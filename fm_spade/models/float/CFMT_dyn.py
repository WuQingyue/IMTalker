from turtle import pos
import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.FMT_dyn import FlowMatchingTransformer
	
class ConditionFMT(BaseModel):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_frames_for_clip = self.opt.num_now_frames
        self.num_prev_frames = self.opt.num_prev_frames
        self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames
        self.audio_encoder = AudioEncoder(opt)
        self.audio_input_dim = 768 if opt.only_last_features else 12 * 768
        self.audio_projection = nn.Sequential(
            nn.Linear(self.audio_input_dim, opt.dim_c),
            nn.LayerNorm(opt.dim_c),
            nn.SiLU()
        )
        # FMT; Flow Matching Transformer
        self.fmt = FlowMatchingTransformer(opt)
        self.odeint_kwargs = {
            'atol': self.opt.ode_atol,
            'rtol': self.opt.ode_rtol,
            'method': self.opt.torchdiffeq_ode_method
        }

    def forward(self, batch, t):
        # 直接从 batch 获取处理好的数据
        # 不再需要 m_now, m_prev, a_now, a_prev
        m_full, a_full, m_ref = batch["m_full"], batch["a_full"], batch["m_ref"]
        a_proj = self.audio_projection(a_full)
        batch["a_full"] = a_proj  # 更新 batch 供底层模型使用

        # 调用底层模型
        # self.fmt.forward 也需要调整以接收新的 batch 格式
        pred_full = self.fmt(batch, t)

        # 重点：这里不再切片！
        # 模型直接返回对完整序列的预测
        return pred_full

    @torch.no_grad()
    def sample(
        self,
        data: dict,
        a_cfg_scale: float = 1.0,
        nfe: int = 10,
        seed: int = None,
        min_chunk_len: int = 10,
        min_prev_num_frames: int = 5
    ) -> torch.Tensor:

        a, ref_x = data['a'], data['ref_x']
        B = a.shape[0]
        # make time
        time = torch.linspace(0, 1, nfe, device=self.opt.rank)

        # encoding audio first with whole audio
        a = a.to(self.opt.rank)
        L = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)  # 总帧数
        a = self.audio_encoder.inference(a, seq_len=L)
        a = self.audio_projection(a)
        chunks = []
        generated_frames = 0

        # 切 chunk 规则
        while generated_frames < L:
            remaining = L - generated_frames
            if remaining < min_chunk_len:
                chunks[-1]['len'] += remaining
                break
            elif remaining < self.num_frames_for_clip:
                chunk_len = remaining
            else:
                chunk_len = self.num_frames_for_clip

            # 计算前文帧数
            if generated_frames >= self.num_prev_frames:
                prev_frames = self.num_prev_frames
            elif generated_frames > min_prev_num_frames:
                prev_frames = generated_frames
            else:
                prev_frames = min_prev_num_frames

            chunks.append({
                "start": max(0, generated_frames - prev_frames),
                "len": chunk_len,
                "prev_frames": prev_frames
            })
            generated_frames += chunk_len

        sample = []
        sample_t = None  # 存放上一步生成结果

        # 分 chunk 生成
        for ci, chunk in enumerate(chunks):
            chunk_start = chunk["start"]
            chunk_len = chunk["len"]
            prev_frames = chunk["prev_frames"]

            # 取前文
            if ci == 0:
                prev_x_t = torch.zeros(B, prev_frames, self.opt.dim_c).to(self.opt.rank)
                prev_a_t = torch.zeros(B, prev_frames, self.opt.dim_w).to(self.opt.rank)
            else:
                prev_x_data = sample_t[:, -prev_frames:] if sample_t is not None else torch.zeros(B, prev_frames, self.opt.dim_c).to(self.opt.rank)
                prev_a_data = a_t[:, -prev_frames:]
                if prev_x_data.shape[1] < prev_frames:  # 补 0
                    pad_len = prev_frames - prev_x_data.shape[1]
                    prev_x_pad = torch.zeros(B, pad_len, self.opt.dim_c).to(self.opt.rank)
                    prev_a_pad = torch.zeros(B, pad_len, self.opt.dim_w).to(self.opt.rank)
                    prev_x_data = torch.cat([prev_x_pad, prev_x_data], dim=1)
                    prev_a_data = torch.cat([prev_a_pad, prev_a_data], dim=1)
                prev_x_t = prev_x_data
                prev_a_t = prev_a_data

            # 当前音频特征
            a_t = a[:, chunk_start:chunk_start + chunk_len]
            if a_t.shape[1] < chunk_len:  # padding by replicate
                a_t = F.pad(a_t, (0, 0, 0, chunk_len - a_t.shape[1]), mode='replicate')

            # 初始化随机噪声
            if self.opt.fix_noise_seed:
                seed_val = self.opt.seed if seed is None else seed
                g = torch.Generator(self.opt.rank).manual_seed(seed_val)
                x0 = torch.randn(B, chunk_len, self.opt.dim_w, device=self.opt.rank, generator=g)
            else:
                x0 = torch.randn(B, chunk_len, self.opt.dim_w, device=self.opt.rank)

            # 定义采样函数
            def sample_chunk(tt, zt):
                out = self.fmt.forward_with_cfv(
                    t=tt.unsqueeze(0),
                    x=zt,
                    a=a_t,
                    prev_x=prev_x_t,
                    prev_a=prev_a_t,
                    ref_x=ref_x,
                    a_cfg_scale=a_cfg_scale,
                )
                out_current = out[:, prev_frames:]  # 去掉前文部分
                return out_current

            # ODE求解
            trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
            sample_t = trajectory_t[-1]

            sample.append(sample_t)

        # 拼接所有 chunk
        sample = torch.cat(sample, dim=1)[:, :L]
        return sample

################# Condition Encoders ################
class AudioEncoder(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.only_last_features = opt.only_last_features
		
		self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(opt.num_prev_frames)

		self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
		self.wav2vec2.feature_extractor._freeze_parameters()

		for name, param in self.wav2vec2.named_parameters():
			param.requires_grad = False



	def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
		if self.only_last_features:
			a = a.last_hidden_state
		else:
			a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
			a = a.reshape(a.shape[0], a.shape[1], -1)
		return a

	def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
		if prev_a is not None:
			a = torch.cat([prev_a, a], dim = 1)
			if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
		else:
			if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
	
		return a

	@torch.no_grad()
	def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
			a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
		a = self.get_wav2vec2_feature(a, seq_len=seq_len)
		return a