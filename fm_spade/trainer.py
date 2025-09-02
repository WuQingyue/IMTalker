from torch.optim import Adam
from accelerate import Accelerator
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from rectify_flow import RectifiedFlow
from dataset.dataset import AudioMotionDataset
from torch.nn import Module, ModuleList
from pathlib import Path
import math
import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint
import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.tensorboard import SummaryWriter  # 确保导入SummaryWriter
from tqdm import tqdm  # 新增进度条库
from datetime import datetime  # 新增时间戳功能

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# tensor helpers

def append_dims(t, ndims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * ndims))

# normalizing helpers

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# noise schedules

def cosmap(t):
    # Algorithm 21 in https://arxiv.org/abs/2403.03206
    return 1. - (1. / (torch.tan(pi / 2 * t) + 1))
def divisible_by(num, den):
    return (num % den) == 0

# losses

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        rectified_flow: dict | RectifiedFlow,
        *,
        dataset: dict,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        batch_size = 16,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 100000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = AudioMotionDataset(**dataset)

        if isinstance(rectified_flow, dict):
            rectified_flow = RectifiedFlow(**rectified_flow)

        self.model = rectified_flow

        # determine whether to keep track of EMA (if not using consistency FM)
        # which will determine which model to use for sampling

        use_ema &= not getattr(self.model, 'use_consistency', False)

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # optimizer, dataloader, and all that

        self.optimizer = Adam(rectified_flow.parameters(), lr = learning_rate, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        self.return_loss_breakdown = isinstance(rectified_flow, RectifiedFlow)

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()
        log_dir = f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.accelerator.unwrap_model(self.optimizer).state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return
        
        load_package = torch.load(path)
        
        self.model.load_state_dict(load_package["model"])
        self.ema_model.load_state_dict(load_package["ema_model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)


    def forward(self):
        dl = cycle(self.dl)
        pbar = tqdm(total=self.num_train_steps, desc='Training', dynamic_ncols=True,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            data = next(dl)

            if self.return_loss_breakdown:
                loss, loss_breakdown = self.model(data, return_loss_breakdown = True)
                self.log(loss_breakdown._asdict(), step = step)
            else:
                loss = self.model(data)

            #self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if getattr(self.model, 'use_consistency', False):
                self.model.ema_model.update()

            if self.is_main and self.use_ema:
                self.ema_model.ema_model.data_shape = self.model.data_shape
                self.ema_model.update()

            self.accelerator.wait_for_everyone()
            if self.is_main and divisible_by(step, self.checkpoint_every):
                self.save(f'checkpoint.{step}.pt')

            # 更新TensorBoard日志
            self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'step': self.global_step
            })
            pbar.update(1)
            self.global_step += 1
                
        pbar.close()
        self.writer.close()
        print('training complete')
