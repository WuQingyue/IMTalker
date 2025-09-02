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
from traitlets import default
from dataset.dataset import DynamicAudioMotionDataset
from models.float.CFMT_dyn import ConditionFMT
from options.base_options import BaseOptions
from pytorch_lightning.loggers import TensorBoardLogger

# Helpers
def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def cosmap(t):
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))
class L1loss(Module):
    def forward(self, pred, target, batch):
        """
        计算带掩码的 L1 Loss，只在 'now' 部分的有效帧上计算。
        每帧取特征维度平均，避免 loss 数值过大。
        """
        split_points = batch["split_points"]  # [B]
        total_lens = batch["m_full_lens"]     # [B]
        B, T_padded, D = pred.shape

        arange = torch.arange(T_padded, device=pred.device).unsqueeze(0).expand(B, -1)
        loss_mask = (arange >= split_points.unsqueeze(1)) & (arange < total_lens.unsqueeze(1))
        loss_mask = loss_mask.unsqueeze(-1)  # [B, T_padded, 1]

        loss_unreduced = torch.abs(pred - target) * loss_mask
        num_valid_elements = loss_mask.sum()

        if num_valid_elements > 0:
            return loss_unreduced.sum() / (num_valid_elements * D)
        else:
            return torch.tensor(0.0, device=pred.device)
class System(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.model = ConditionFMT(opt)
        self.opt = opt
        self.loss_fn = L1loss()

    def forward(self, x):
        return self.model(x)
    
    # ---------------- vvv 最终正确的训练和验证逻辑 vvv ------------------
    def _shared_step(self, batch):
        """将训练和验证的通用逻辑提取出来"""

        # 1. 解包数据，并保留一份干净的 m_full 用于计算 gt_flow
        m_full_gt = batch["m_full"]
        B, T_padded, D = m_full_gt.shape
        
        # 2. 创建一个只标识 "now" 部分的掩码
        split_points = batch["split_points"]
        total_lens = batch["m_full_lens"]
        arange = torch.arange(T_padded, device=m_full_gt.device).unsqueeze(0).expand(B, -1)
        now_mask = (arange >= split_points.unsqueeze(1)) & (arange < total_lens.unsqueeze(1))
        now_mask = now_mask.unsqueeze(-1) # Shape: [B, T_padded, 1]

        # 3. 只对 "now" 部分加噪
        noise = torch.randn_like(m_full_gt)
        times = torch.rand(B, device=self.device)
        t_expanded = append_dims(times, m_full_gt.ndim - 1)
        
        # 计算理论上全加噪的张量
        fully_noised_tensor = t_expanded * m_full_gt + (1 - t_expanded) * noise
        
        # 使用 torch.where 根据掩码选择：
        # - 在 "now" 部分 (mask=True)，使用加噪数据
        # - 在 "prev" 部分 (mask=False)，使用原始干净数据
        noised_input = torch.where(now_mask, fully_noised_tensor, m_full_gt)
        
        # ground truth flow 是 m_full - noise。损失函数会自动用掩码处理，只关心 now 部分。
        gt_flow = m_full_gt - noise

        # 4. 准备模型输入并预测
        batch["m_full"] = noised_input
        pred_flow = self.model(batch, t=times)

        # 5. 计算损失 (loss_fn 内部会使用掩码)
        fm_loss = self.loss_fn(pred_flow, gt_flow, batch)
        
        # 6. 计算速度损失 (loss_fn 内部同样会使用掩码)
        pred_vel = pred_flow[:, 1:] - pred_flow[:, :-1]
        gt_vel = gt_flow[:, 1:] - gt_flow[:, :-1]
        vel_batch = batch.copy()
        vel_batch["m_full_lens"] = torch.clamp(batch["m_full_lens"] - 1, min=0)
        # 速度掩码需要特殊处理，确保不跨越 prev/now 边界
        # 但我们的 loss_fn 内部处理得很好，它会用新的 m_full_lens 和旧的 split_points，
        # 自动忽略掉边界和 prev 部分。
        velocity_loss = self.loss_fn(pred_vel, gt_vel, vel_batch)
        
        return fm_loss, velocity_loss

    def training_step(self, batch, batch_idx):
        fm_loss, velocity_loss = self._shared_step(batch)
        
        train_loss = fm_loss + velocity_loss
        self.log("train/loss", train_loss, prog_bar=True)
        self.log("train/fm_loss", fm_loss)
        self.log("train/vel_loss", velocity_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        fm_loss, velocity_loss = self._shared_step(batch)
        
        val_loss = fm_loss + velocity_loss
        self.log("val/loss", val_loss, prog_bar=True)
        self.log("val/fm_loss", fm_loss)
        self.log("val/vel_loss", velocity_loss)
    
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
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--iter', default=5000000, type=int)
        parser.add_argument('--ckpt_path', default="./checkpoints/float.pth", type=str)
        parser.add_argument("--exp_path", type=str, default='./exps')
        parser.add_argument("--exp_name", type=str, default='debug')
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        parser.add_argument("--dim_motion", type=int, default=512)
        parser.add_argument("--dim_c", type=int, default=512)
        parser.add_argument("--min_prev_len", type =int, default=10)
        parser.add_argument("--min_now_len", type =int, default=10)
        parser.add_argument("--max_seq_len", type =int, default=250)
        
        
        return parser

class DataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):
        # 确保使用我们修改后的 DynamicAudioMotionDataset
        self.train_dataset = DynamicAudioMotionDataset(opt=self.opt, start=100, end=None)
        self.val_dataset = DynamicAudioMotionDataset(opt=self.opt, start=0, end=100)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset, 
            num_workers=8, 
            batch_size=self.opt.batch_size, 
            shuffle=True, 
            persistent_workers=True,
            collate_fn=DynamicAudioMotionDataset.collate_fn # <-- 必须添加这一行
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset, 
            num_workers=0, 
            batch_size=8, 
            shuffle=False,
            collate_fn=DynamicAudioMotionDataset.collate_fn # <-- 必须添加这一行
        )

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