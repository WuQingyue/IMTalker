import argparse
import os
import torch
from torch.utils import data
from dataset_512 import VFHQ_mask_neg, CombinedDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import torch
import torch.nn.functional as F
from torch import nn, optim
from networks.model_bestdecoder import IMFModel
from vgg19_mask import VGGLoss_mask
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from arcface import FaceSimLoss, Backbone

# 在IMFSystem类中添加加载方法
class IMFSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(vars(args))
        self.args = args
        self.gen = IMFModel(args)
        
        if args.use_gan:
            from networks.discriminator import PatchDiscriminator
            self.disc = PatchDiscriminator()
        
        if args.use_arcface:
            arcface = Backbone(50, 0.65, mode='ir_se')
            arcface.load_state_dict(torch.load(args.arcface_path))
            arcface.cuda()
            self.face_loss_fn = FaceSimLoss(arcface_model=arcface)

        self.criterion_vgg = VGGLoss_mask()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # 获取优化器
        if self.args.use_gan:
            opt_g, opt_d = self.optimizers()
        else:
            opt_g = self.optimizers()
            opt_d = None
    
        real = batch["image_1_512"]
        
        # ===================================================================
        #                 步骤 1: 训练判别器 (Train Discriminator)
        # ===================================================================
        if self.args.use_gan:
            opt_d.zero_grad()
            
            # 生成假图片，并用 detach() 防止梯度流回生成器
            with torch.no_grad():
                fake = self.gen(batch["image_1_256"], batch["image_0_256"]).detach()
    
            # 判别器对真假图片进行预测
            pred_real = self.disc(real)
            pred_fake = self.disc(fake)
            
            # 计算判别器损失
            loss_d = self.calculate_gan_loss(pred_real, pred_fake, is_generator=False)
            
            # 计算并加入 R1 正则化损失 (如果启用)
            r1_loss = torch.tensor(0.0, device=self.device)
            if self.args.use_r1_reg and self.global_step > 0 and self.global_step % self.args.r1_reg_every == 0:
                real.requires_grad = True
                pred_real_for_reg = self.disc(real)
                r1_loss = self.r1_regularization(pred_real_for_reg, real)
                loss_d += r1_loss  # 将R1损失加入判别器总损失
                real.requires_grad = False # 及时关闭梯度的需求
    
            # 反向传播并更新判别器
            self.manual_backward(loss_d)
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=1.0)
            opt_d.step()
    
        # ===================================================================
        #                 步骤 2: 训练生成器 (Train Generator)
        # ===================================================================
        opt_g.zero_grad()
        
        # 生成器前向传播
        #pred = self.gen(batch["image_1_256"], batch["image_0_256"])
        f_r,f = self.gen.dense_feature_encoder(batch["image_0_256"])
        t_r = self.gen.latent_token_encoder(batch["image_0_256"])
        t_c = self.gen.latent_token_encoder(batch["image_1_256"])
        t_neg = self.gen.latent_token_encoder(batch["neg_image_256"])
        pred = self.gen.decode_latent_tokens(f_r, t_r, t_c, f)
        pred_neg = self.gen.decode_latent_tokens(f_r, t_r, t_neg, f)
        pred_neg_256 = F.interpolate(pred_neg, size=(256, 256), mode='bilinear', align_corners=False)
        neg_f = self.gen.encode_dense_feature(pred_neg_256)

        # 计算重建损失 (L1, VGG)
        l1_loss = F.l1_loss(pred, real)
        vgg_loss_all, vgg_loss_face = self.criterion_vgg(pred, real, batch["mask_eye_1"] + batch["mask_mouth_1"])
        
        # 初始化生成器总损失
        total_g_loss = self.args.loss_l1 * l1_loss + self.args.loss_vgg_all * vgg_loss_all + self.args.loss_vgg_face * vgg_loss_face
        
        neg_l1_loss = F.l1_loss(f, neg_f)

        total_g_loss += neg_l1_loss
        # 计算生成器的对抗损失 (如果启用)
        loss_g_gan = torch.tensor(0.0, device=self.device)
        if self.args.use_gan:
            # 此时的 pred (fake) 不能 detach，因为我们需要梯度流回生成器
            pred_fake_for_g = self.disc(pred)
            loss_g_gan = self.calculate_gan_loss(None, pred_fake_for_g, is_generator=True)
            total_g_loss += self.args.gan_weight * loss_g_gan
        
        if self.args.use_arcface:
            loss_face = self.face_loss_fn(pred_neg, real)
            total_g_loss += self.args.loss_arcface * loss_face
        
        # 反向传播并更新生成器
        self.manual_backward(total_g_loss)
        torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0)
        opt_g.step()
    
        # ===================================================================
        #                         步骤 3: 日志记录
        # ===================================================================
        log_dict = {
            'train/g_total_loss': total_g_loss, # 记录生成器的总损失
            'train/l1_loss': l1_loss.detach(),
            'train/vgg_loss_all': vgg_loss_all.detach(),
            'train/vgg_loss_face': vgg_loss_face.detach(),
            'train/neg_l1_loss': neg_l1_loss.detach(),
        }
        if self.args.use_gan:
            log_dict.update({
                'train/g_gan_loss': loss_g_gan.detach(),
                'train/d_loss': loss_d.detach(),
                'train/r1_loss': r1_loss.detach()
            })
        if self.args.use_arcface:
            log_dict.update({
                'train/arcface_loss': loss_face.detach()
            })
        self.log_dict(log_dict, prog_bar=True)
        
        # 返回生成器的总损失，因为它通常是主要的监控指标
        return total_g_loss
    
    def validation_step(self, batch, batch_idx):
        # --------------------------
        # 统一前向传播流程
        # --------------------------
        # 提取 dense feature 和 latent token
        f_r, f = self.gen.dense_feature_encoder(batch["image_0_256"])
        t_r = self.gen.latent_token_encoder(batch["image_0_256"])
        t_c = self.gen.latent_token_encoder(batch["image_1_256"])
        t_neg = self.gen.latent_token_encoder(batch["neg_image_256"])
    
        # 正样本生成
        pred = self.gen.decode_latent_tokens(f_r, t_r, t_c, f)
        # 负样本生成
        pred_neg = self.gen.decode_latent_tokens(f_r, t_r, t_neg, f)
        pred_neg_256 = F.interpolate(pred_neg, size=(256, 256), mode='bilinear', align_corners=False)
        neg_f = self.gen.encode_dense_feature(pred_neg_256)
    
        # 重建 (self reconstruction)
        recon = self.gen.decode_latent_tokens(f_r, t_r, t_r, f)
    
        # --------------------------
        # 损失计算
        # --------------------------
        mask_1 = batch["mask_eye_1"] + batch["mask_mouth_1"]
        mask_0 = batch["mask_eye_0"] + batch["mask_mouth_0"]
    
        # 正向损失
        pred_l1 = F.l1_loss(pred, batch["image_1_512"])
        pred_vgg_all, pred_vgg_face = self.criterion_vgg(pred, batch["image_1_512"], mask_1)
    
        # 重建损失
        recon_l1 = F.l1_loss(recon, batch["image_0_512"])
        recon_vgg_all, recon_vgg_face = self.criterion_vgg(recon, batch["image_0_512"], mask_0)
    
        # 负样本 L1 loss
        neg_l1_loss = F.l1_loss(f, neg_f)
    
        # 总损失
        loss_pred = self.hparams.loss_l1 * pred_l1 + \
                    self.hparams.loss_vgg_all * pred_vgg_all + \
                    self.hparams.loss_vgg_face * pred_vgg_face
    
        loss_recon = self.hparams.loss_l1 * recon_l1 + \
                     self.hparams.loss_vgg_all * recon_vgg_all + \
                     self.hparams.loss_vgg_face * recon_vgg_face
    
        val_loss = (loss_pred + loss_recon) / 2 + neg_l1_loss
    
        # ArcFace loss
        if self.args.use_arcface:
            loss_face = self.face_loss_fn(pred_neg, batch["image_1_512"])
            val_loss += self.args.loss_arcface * loss_face
    
        # --------------------------
        # 日志记录
        # --------------------------
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/pred_loss", loss_pred, prog_bar=False, sync_dist=True)
        self.log("val/recon_loss", loss_recon, prog_bar=False, sync_dist=True)
        self.log("val/neg_l1_loss", neg_l1_loss, prog_bar=False, sync_dist=True)
        if self.args.use_arcface:
            self.log("val/arcface_loss", loss_face, prog_bar=False, sync_dist=True)
    
        # 可视化图像
        name_list = ['input_0', 'input_1', 'pred', 'recon', 'mask_0', 'mask_1', 'neg_pred']
        img_list = [
            batch["image_0_512"], batch["image_1_512"],
            pred, recon,
            mask_0, mask_1,
            pred_neg
        ]
        for name, img in zip(name_list, img_list):
            self.logger.experiment.add_images(
                tag=name,
                img_tensor=img,
                global_step=self.global_step
            )
    
        return val_loss

    
    def calculate_gan_loss(self, pred_real, pred_fake, is_generator):
        if is_generator:
            return sum([F.softplus(-pred).mean() for pred in pred_fake])
        real_loss = sum([F.softplus(-real).mean() for real in pred_real])
        fake_loss = sum([F.softplus(fake).mean() for fake in pred_fake])
        return (real_loss + fake_loss)

    def configure_optimizers(self):
        opt_g = optim.Adam(self.gen.parameters(), 
                           lr=self.args.lr, 
                           betas=(0.5, 0.999))
        scheduler_g = {
        'scheduler': optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.args.iter, eta_min=1e-5),
        'interval': 'step', # 明确指定按步更新
        }
        if self.args.use_gan:
            opt_d = optim.Adam(self.disc.parameters(),
                               lr=self.args.lr * self.args.gan_weight,
                               betas=(0.5, 0.999))
            scheduler_d = {
                'scheduler': optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.args.iter, eta_min=1e-5),
                'interval': 'step',
                }
            return [opt_g, opt_d], [scheduler_g, scheduler_d]
        return {"optimizer": opt_g, "lr_scheduler": scheduler_g}

    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]

        # ------------------- 加载 Generator 权重 -------------------
        # 1. 从ckpt中筛选出generator的权重
        ckpt_gen_state_dict = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
        
        # 2. 创建一个空的字典，用于存放匹配的权重
        safe_gen_state_dict = {}
        
        # 3. 遍历ckpt中的权重，进行筛选
        for key, value in ckpt_gen_state_dict.items():
            # 检查当前模型是否有同名层
            if key in self.gen.state_dict():
                # 检查形状是否一致
                if self.gen.state_dict()[key].shape == value.shape:
                    safe_gen_state_dict[key] = value
                else:
                    print(f"[WARN] Skipped loading '{key}' for generator: shape mismatch. "
                          f"Checkpoint shape: {value.shape}, Model shape: {self.gen.state_dict()[key].shape}")
            # 如果模型中不存在该key，则自动忽略，无需处理
        
        # 4. 使用筛选后的安全字典加载权重
        missing_gen, unexpected_gen = self.gen.load_state_dict(safe_gen_state_dict, strict=False)
        print(f"[INFO] Loaded generator weights.")
        if missing_gen:
            print(f"       - Missing keys: {missing_gen}")
        if unexpected_gen:
            # `unexpected_gen` 理论上应该为空，因为我们已经筛选过了
            print(f"       - Unexpected keys: {unexpected_gen}")

        # ------------------- 加载 Discriminator 权重 -------------------
        if self.args.use_gan and hasattr(self, "disc"):
            # 1. 从ckpt中筛选出discriminator的权重
            ckpt_disc_state_dict = {k.replace("disc.", ""): v for k, v in state_dict.items() if k.startswith("disc.")}
            
            if not ckpt_disc_state_dict:
                print("[WARN] No discriminator weights found in checkpoint.")
                return

            # 2. 创建安全字典
            safe_disc_state_dict = {}

            # 3. 筛选权重
            for key, value in ckpt_disc_state_dict.items():
                if key in self.disc.state_dict():
                    if self.disc.state_dict()[key].shape == value.shape:
                        safe_disc_state_dict[key] = value
                    else:
                        print(f"[WARN] Skipped loading '{key}' for discriminator: shape mismatch. "
                              f"Checkpoint shape: {value.shape}, Model shape: {self.disc.state_dict()[key].shape}")

            # 4. 加载权重
            missing_disc, unexpected_disc = self.disc.load_state_dict(safe_disc_state_dict, strict=False)
            print(f"[INFO] Loaded discriminator weights.")
            if missing_disc:
                print(f"       - Missing keys: {missing_disc}")
            if unexpected_disc:
                print(f"       - Unexpected keys: {unexpected_disc}")

    def r1_regularization(self, pred_real, real_img, r1_gamma=10):
        """计算R1正则化损失"""
        grad_real, = torch.autograd.grad(
            outputs=sum(torch.mean(out) for out in pred_real), # 对所有尺度的输出求和
            inputs=real_img,
            create_graph=True,
        )
        grad_real_penalty = (grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean())
        r1_loss = grad_real_penalty * (r1_gamma / 2)
        return r1_loss


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def setup(self, stage):
        self.train_dataset = CombinedDataset((VFHQ_mask_neg('train', "vfhq"), VFHQ_mask_neg('train', "vox2"),VFHQ_mask_neg('train', "multitalk")),(0.4,0.3,0.3))
        self.val_dataset = CombinedDataset((VFHQ_mask_neg('test', "vfhq"), VFHQ_mask_neg('test', "vox2"),VFHQ_mask_neg('test', "multitalk")),(0.4,0.3,0.3))
        
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=8,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
            shuffle=False
        )

if __name__ == "__main__":
    # Remove the duplicate parser definition and modify like this:
    parser = argparse.ArgumentParser()
    # 添加GAN相关参数
    parser.add_argument("--use_gan", action='store_true', help="Enable GAN training")
    parser.add_argument("--gan_weight", type=float, default=1, help="Weight for GAN loss")
    parser.add_argument("--iter", type=int, default=7000000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--exp_path", type=str, default='./exps')
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_l1", type=float, default=1.0)
    parser.add_argument("--loss_vgg_all", type=float, default=10.0)
    parser.add_argument("--loss_vgg_face", type=float, default=100.0)
    parser.add_argument("--r1_reg_every", type=int, default=16, 
                        help="Frequency to apply R1 regularization (e.g., apply every 16 steps)")
    parser.add_argument("--use_r1_reg", action='store_true', help="gan R1 reg")
    parser.add_argument("--use_arcface", action='store_true', help="use arcface loss")
    parser.add_argument("--arcface_path", type=str, default="E:\codes\codes\model_ir_se50.pth", help="2d position encoding")
    parser.add_argument("--loss_arcface", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)

    
    # 删除下面这行
    # parser = pl.Trainer.add_argparse_args(parser)
    
    # 直接解析参数
    args = parser.parse_args()

    # 训练器参数直接在Trainer初始化时设置
    # 初始化系统
    system = IMFSystem(args)
    dm = CombinedDataModule(args)
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        system.load_ckpt(args.resume_ckpt)
    # 配置logger和checkpoint
    logger = TensorBoardLogger(save_dir=args.exp_path, name=args.exp_name)
    
    # 修改checkpoint回调配置
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.exp_path, args.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=args.save_freq,
        save_top_k=-1,
        save_last=True
    )
    
    # 修改Trainer配置
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=args.iter,
        check_val_every_n_epoch=None,
        val_check_interval=args.display_freq,
        logger=logger,
        callbacks=[
            checkpoint_callback,
        ],
        enable_progress_bar=True,
    )
    trainer.fit(system, dm)



