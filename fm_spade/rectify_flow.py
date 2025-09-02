from __future__ import annotations

import math
from copy import deepcopy
from collections import namedtuple
from typing import Tuple, List, Literal, Callable

import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torchdiffeq import odeint

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange

from ema_pytorch import EMA
from scipy.optimize import linear_sum_assignment
from models.float.FLOAT import FLOAT

# helpers

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

# losses

class PseudoHuberLoss(Module):
    def __init__(self, data_dim: int = 3):
        super().__init__()
        self.data_dim = data_dim

    def forward(self, pred, target, reduction = 'mean', **kwargs):
        data_dim = default(self.data_dim, kwargs.pop('data_dim', None))

        c = .00054 * self.data_dim
        loss = (F.mse_loss(pred, target, reduction = reduction) + c * c).sqrt() - c

        if reduction == 'none':
            loss = reduce(loss, 'b ... -> b', 'mean')

        return loss


class MSELoss(Module):
    def forward(self, pred, target, **kwargs):
        return F.mse_loss(pred, target)

# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', ['total', 'main', 'data_match', 'velocity_match'])

# main class

class RectifiedFlow(Module):
    def __init__(
        self,
        model: dict | Module,
        time_cond_kwarg: str | None = 'times',
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
        predict: Literal['flow', 'noise'] = 'flow',
        loss_fn: Literal[
            'mse',
            'pseudo_huber',
            'pseudo_huber_with_lpips'
        ] | Module = 'mse',
        noise_schedule: Literal[
            'cosmap'
        ] | Callable = identity,
        loss_fn_kwargs: dict = dict(),
        ema_update_after_step: int = 100,
        ema_kwargs: dict = dict(),
        data_shape: Tuple[int, ...] | None = None,
        immiscible = False,
        use_consistency = False,
        consistency_decay = 0.9999,
        consistency_velocity_match_alpha = 1e-5,
        consistency_delta_time = 1e-3,
        consistency_loss_weight = 1.,
        data_normalize_fn = normalize_to_neg_one_to_one,
        data_unnormalize_fn = unnormalize_to_zero_to_one,
        clip_during_sampling = False,
        clip_values: Tuple[float, float] = (-1., 1.),
        clip_flow_during_sampling = None, # this seems to help a lot when training with predict epsilon, at least for me
        clip_flow_values: Tuple[float, float] = (-3., 3)
    ):
        super().__init__()

        self.model = model
        self.time_cond_kwarg = time_cond_kwarg # whether the model is to be conditioned on the times

        # objective - either flow or noise (proposed by Esser / Rombach et al in SD3)

        self.predict = predict

        # automatically default to a working setting for predict epsilon

        clip_flow_during_sampling = default(clip_flow_during_sampling, predict == 'noise')

        # loss fn

        if loss_fn == 'mse':
            loss_fn = MSELoss()

        elif loss_fn == 'pseudo_huber':
            assert predict == 'flow'

            # section 4.2 of https://arxiv.org/abs/2405.20320v1
            loss_fn = PseudoHuberLoss(**loss_fn_kwargs)


        elif not isinstance(loss_fn, Module):
            raise ValueError(f'unknown loss function {loss_fn}')

        self.loss_fn = loss_fn

        # noise schedules

        if noise_schedule == 'cosmap':
            noise_schedule = cosmap

        elif not callable(noise_schedule):
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        self.noise_schedule = noise_schedule

        # sampling

        self.odeint_kwargs = odeint_kwargs
        self.data_shape = data_shape

        # clipping for epsilon prediction

        self.clip_during_sampling = clip_during_sampling
        self.clip_flow_during_sampling = clip_flow_during_sampling

        self.clip_values = clip_values
        self.clip_flow_values = clip_flow_values

        # consistency flow matching

        self.use_consistency = use_consistency
        self.consistency_decay = consistency_decay
        self.consistency_velocity_match_alpha = consistency_velocity_match_alpha
        self.consistency_delta_time = consistency_delta_time
        self.consistency_loss_weight = consistency_loss_weight

        if use_consistency:
            self.ema_model = EMA(
                model,
                beta = consistency_decay,
                update_after_step = ema_update_after_step,
                include_online_model = False,
                **ema_kwargs
            )

        # immiscible diffusion paper, will be removed if does not work

        self.immiscible = immiscible

        # normalizing fn

        self.data_normalize_fn = default(data_normalize_fn, identity)
        self.data_unnormalize_fn = default(data_unnormalize_fn, identity)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict_flow(self, model: Module, batch, *, times, eps=1e-10):
        batch_size = batch['m_now'].shape[0]
    
        model_kwargs = {}
        time_kwarg = self.time_cond_kwarg
    
        if exists(time_kwarg):
            times = rearrange(times, '... -> (...)')
            if times.numel() == 1:
                times = repeat(times, '1 -> b', b=batch_size)
    
        output = model(batch, t=times)
    
        if self.predict == 'flow':
            flow = output
        else:
            raise ValueError(f'unknown objective {self.predict}')
    
        return output, flow
    
    
    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        r_cfg_scale: float = 1.0,
        e_cfg_scale: float = 1.0,
        emo: str = None,
        nfe: int = 10,
        seed: int = None
    ) -> torch.Tensor:
    
        r_s, a = data['r_s'], data['a']
        B = a.shape[0]
    
        time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)
    
        a = a.to(self.opt.rank)
        T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
        wa = self.audio_encoder.inference(a, seq_len=T)
    
        emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
        if emo_idx is None:
            we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
        else:
            we = F.one_hot(torch.tensor(emo_idx, device=a.device), num_classes=self.opt.dim_e).unsqueeze(0).unsqueeze(0)
    
        sample = []
        for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
            if self.opt.fix_noise_seed:
                seed = self.opt.seed if seed is None else seed
                g = torch.Generator(self.opt.rank)
                g.manual_seed(seed)
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank, generator=g)
            else:
                x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device=self.opt.rank)
    
            if t == 0:
                prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
                prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
            else:
                prev_x_t = sample_t[:, -self.num_prev_frames:]
                prev_wa_t = wa_t[:, -self.num_prev_frames:]
    
            wa_t = wa[:, t * self.num_frames_for_clip: (t + 1) * self.num_frames_for_clip]
    
            if wa_t.shape[1] < self.num_frames_for_clip:
                wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')
    
            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfv(
                    t=tt.unsqueeze(0),
                    x=zt,
                    wa=wa_t,
                    wr=r_s,
                    we=we,
                    prev_x=prev_x_t,
                    prev_wa=prev_wa_t,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale
                )
                out_current = out[:, self.num_prev_frames:]
                return out_current
    
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]
            sample.append(sample_t)
    
        sample = torch.cat(sample, dim=1)[:, :T]
        return sample
    
    
    def forward(
        self,
        data: dict,
        noise: Tensor | None = None,
        return_loss_breakdown=False,
        **model_kwargs
    ):
        batch = data
        a_prev, m_prev, a_now, m_now = data["a_prev"], data["m_prev"], data["a_now"], data["m_now"]
        x = m_now
        batch_size, *data_shape = x.shape
    
        #x = self.data_normalize_fn(x)
        #self.data_shape = default(self.data_shape, data_shape)
    
        noise = default(noise, torch.randn_like(x))
    
#        if self.immiscible:
#            cost = torch.cdist(m_now.flatten(1), noise.flatten(1))
#            _, reorder_indices = linear_sum_assignment(cost.cpu())
#            noise = noise[from_numpy(reorder_indices).to(cost.device)]
    
        times = torch.rand(batch_size, device=self.device)
        padded_times = append_dims(times, x.ndim - 1)
        if self.use_consistency:
            padded_times *= 1. - self.consistency_delta_time
    
        def get_noised_and_flows(model, t):
            t = self.noise_schedule(t)
            noised = t * x + (1. - t) * noise
            flow = x - noise
    
            batch["m_now"] = noised
            model_output, pred_flow = self.predict_flow(model, batch, times=t)
            #print(noised.shape, pred_flow.shape, t.shape)
            pred_data = noised + pred_flow * (1. - t)
    
            return model_output, flow, pred_flow, pred_data
    
        output, flow, pred_flow, pred_data = get_noised_and_flows(self.model, padded_times)
    
        if self.use_consistency:
            delta_t = self.consistency_delta_time
            ema_output, ema_flow, ema_pred_flow, ema_pred_data = get_noised_and_flows(self.ema_model, padded_times + delta_t)
    
        if self.predict == 'flow':
            target = flow
        elif self.predict == 'noise':
            target = noise
        else:
            raise ValueError(f'unknown objective {self.predict}')
    
        main_loss = self.loss_fn(output, target, pred_data=pred_data, times=times, data=x)
    
        consistency_loss = data_match_loss = velocity_match_loss = 0.
        if self.use_consistency:
            data_match_loss = F.mse_loss(pred_data, ema_pred_data)
            velocity_match_loss = F.mse_loss(pred_flow, ema_pred_flow)
            consistency_loss = data_match_loss + velocity_match_loss * self.consistency_velocity_match_alpha
    
        total_loss = main_loss + consistency_loss * self.consistency_loss_weight
    
        if not return_loss_breakdown:
            return total_loss
    
        return total_loss, LossBreakdown(total_loss, main_loss, data_match_loss, velocity_match_loss)
    
    