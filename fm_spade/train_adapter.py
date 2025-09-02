import os
import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import pi
from torch.nn import Module
from torch.utils import data
from torch import nn, optim
from einops import rearrange, repeat
from dataset.dataset import AudioMotionSmirkGazeDataset
from models.float.CFMT_var import ConditionFMT
from options.base_options import BaseOptions
from pytorch_lightning.loggers import TensorBoardLogger

# Helpers
def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def cosmap(t):
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))

class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)
class L1loss(Module):
    def forward(self, pred, target, **kwargs):
        return F.l1_loss(pred, target)
class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = ConditionFMT(opt)
        self.opt = opt
        self.loss_fn = L1loss()

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        """
        包含训练和验证共享逻辑的辅助函数。
        执行：解包 -> 加噪 -> 预测 -> 计算所有损失。
        返回一个包含所有损失项的字典。
        """
        # 1. 解包所有需要的数据
        m_now = batch["m_now"]
        gaze, prev_gaze = batch["gaze_now"], batch["gaze_prev"]
        pose, prev_pose = batch["pose_now"], batch["pose_prev"]
        cam, prev_cam = batch["cam_now"], batch["cam_prev"]

        # 2. 加噪 (逻辑不变)
        noise = torch.randn_like(m_now)
        times = torch.rand(m_now.size(0), device=self.device)
        t_expand = append_dims(times, m_now.ndim - 1)
        noised_motion = t_expand * m_now + (1 - t_expand) * noise
        gt_flow = m_now - noise

        # 3. 准备输入并进行预测 (修正了返回值的解包)
        batch["m_now"] = noised_motion
        pred_flow_anchor, pred_condition = self.model(batch, t=times)

        # 4. 计算所有损失
        # 4.1 Flow Matching Loss
        fm_loss = self.loss_fn(pred_flow_anchor, gt_flow)

        # 4.2 Velocity Loss
        velocity_loss = self.loss_fn(
            pred_flow_anchor[:, 1:] - pred_flow_anchor[:, :-1],
            gt_flow[:, 1:] - gt_flow[:, :-1]
        )

        # 4.3 Adapter Loss
        # 准备 adapter 损失的目标真值 (Ground Truth)
        gaze_full = torch.concat((prev_gaze, gaze), dim=1)
        pose_full = torch.concat((prev_pose, pose), dim=1)
        cam_full = torch.concat((prev_cam, cam), dim=1)
        adapter_gt = torch.concat((gaze_full, pose_full, cam_full), dim=-1)
        adapter_loss = self.loss_fn(pred_condition, adapter_gt)

        # 5. 计算总损失
        total_loss = fm_loss + velocity_loss + adapter_loss

        # 6. 以字典形式返回所有损失，方便后续记录
        return {
            "total_loss": total_loss,
            "fm_loss": fm_loss,
            "velocity_loss": velocity_loss,
            "adapter_loss": adapter_loss
        }

    def training_step(self, batch, batch_idx):
        # 调用共享逻辑
        losses = self._shared_step(batch)

        # 记录训练日志，使用 'train_' 前缀
        self.log("train_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log("train_fm_loss", losses["fm_loss"], sync_dist=True)
        self.log("train_velocity_loss", losses["velocity_loss"], sync_dist=True)
        self.log("train_adapter_loss", losses["adapter_loss"], prog_bar=True, sync_dist=True)

        # 返回总损失以进行反向传播
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        # 调用共享逻辑
        losses = self._shared_step(batch)

        # 记录验证日志，使用 'val_' 前缀
        self.log("val_loss", losses["total_loss"], prog_bar=True, sync_dist=True)
        self.log("val_fm_loss", losses["fm_loss"], sync_dist=True)
        self.log("val_velocity_loss", losses["velocity_loss"], sync_dist=True)
        self.log("val_adapter_loss", losses["adapter_loss"], sync_dist=True)

        # validation_step 不需要返回损失
    
    def load_ckpt(self, ckpt_path):
        print(f"[INFO] Loading weights from checkpoint: {ckpt_path}")

        # 载入 checkpoint 并提取 state_dict
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)  # 兼容直接是 dict 或包含 state_dict 的情况

        # 处理 "model." 前缀（兼容 DDP 保存）
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # 当前模型参数
        model_state_dict = self.model.state_dict()

        # 用于保存能加载的参数
        loadable_params = {}
        unmatched_keys = []

        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                loadable_params[k] = v
            else:
                unmatched_keys.append(k)

        # 加载匹配的权重
        missing_keys, unexpected_keys = self.model.load_state_dict(loadable_params, strict=False)

        print(f"[INFO] Loaded {len(loadable_params)} params from checkpoint.")
        if missing_keys:
            print(f"[WARNING] Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys in checkpoint: {unexpected_keys}")
        if unmatched_keys:
            print(f"[WARNING] {len(unmatched_keys)} keys skipped due to shape mismatch or not found in model:")
            for k in unmatched_keys:
                print(f"  - {k}")

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.opt.iter, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": scheduler}
    
class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument("--audio_path", default=None, type=str)
        parser.add_argument("--motion_path", default=None, type=str)
        parser.add_argument("--smirk_path", default=None, type=str)
        parser.add_argument("--gaze_path", default=None, type=str)
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--iter', default=5000000, type=int)
        parser.add_argument('--ckpt_path', default="./checkpoints/float.pth", type=str)
        parser.add_argument('--imf_path', default="E:\codes\codes\IMF_last\exps\\0.2_vgg\checkpoints\last.ckpt", type=str)
        parser.add_argument("--exp_path", type=str, default='./exps')
        parser.add_argument("--exp_name", type=str, default='debug')
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--dim_motion", type=int, default=512)
        parser.add_argument("--dim_c", type=int, default=512)
        
        
        return parser

class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage):
        self.train_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=100, end=-1)
        self.val_dataset = AudioMotionSmirkGazeDataset(opt=self.opt, start=0, end=100)

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, num_workers=8, batch_size=self.opt.batch_size, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, num_workers=0, batch_size=8, shuffle=False)

if __name__ == '__main__':
    opt = TrainOptions().parse()
    system = System(opt)
    dm = DataModule(opt)

    logger = TensorBoardLogger(save_dir=opt.exp_path, name=opt.exp_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(opt.exp_path, opt.exp_name, 'checkpoints'),
        filename='{step:06d}',
        every_n_train_steps=opt.save_freq,
        save_top_k=-1,
        save_last=True
    )
    if opt.resume_ckpt and os.path.exists(opt.resume_ckpt):
        system.load_ckpt(opt.resume_ckpt)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        strategy='ddp_find_unused_parameters_true' if torch.cuda.device_count() > 1 else 'auto',
        max_steps=opt.iter,
        val_check_interval=opt.display_freq,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(system, dm)