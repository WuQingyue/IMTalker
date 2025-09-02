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
from models.float.FMT import FlowMatchingTransformer

######## Main Phase 2 model ########		
class ConditionFMT(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(self.opt.num_prev_frames)
		self.num_total_frames = self.num_frames_for_clip + self.num_prev_frames
		self.audio_encoder 		= AudioEncoder(opt)
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

	######## Motion Sampling and Inference ########
	def forward(self, batch, t):
		x, prev_x, a, prev_a, m_ref = batch["m_now"], batch["m_prev"], batch["a_now"], batch["a_prev"], batch["m_ref"]
		bs = x.shape[0]
		if not self.opt.only_last_features:
			a = a.reshape(bs, self.num_frames_for_clip, -1)
			prev_a = prev_a.reshape(bs, self.num_prev_frames, -1)
		
		a = self.audio_projection(a)
		prev_a = self.audio_projection(prev_a)
		pred = self.fmt(t, x.squeeze(), a, prev_x, prev_a, m_ref)
		pred = pred[:,self.num_prev_frames:,...]
		return pred

	@torch.no_grad()
	def sample(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		nfe: int = 10,
		seed: int = None
	) -> torch.Tensor:
		
		a, ref_x = data['a'], data['ref_x']
		B = a.shape[0]

		# make time 
		time = torch.linspace(0, 1, nfe, device=self.opt.rank)
		
		# encoding audio first with whole audio
		a = a.to(self.opt.rank)
		T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		a = self.audio_encoder.inference(a, seq_len=T)
		a = self.audio_projection(a)

		sample = []	
		# sampleing chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				seed = self.opt.seed if seed is None else seed	
				g = torch.Generator(self.opt.rank)
				g.manual_seed(seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

			if t == 0: # should define the previous
				prev_x_t = ref_x.repeat(B, self.num_prev_frames, 1).to(self.opt.rank)
				#prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_c).to(self.opt.rank)
				prev_a_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_a_t = a_t[:, -self.num_prev_frames:]
			
			a_t = a[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]
			#a_t = a[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]
			if a_t.shape[1] < self.num_frames_for_clip: # padding by replicate
				a_t = F.pad(a_t, (0, 0, 0, self.num_frames_for_clip - a_t.shape[1]), mode='replicate')
			def sample_chunk(tt, zt):
				out = self.fmt.forward_with_cfv(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						a 			= a_t, 
						prev_x      = prev_x_t,
						prev_a      = prev_a_t,
						ref_x       = ref_x,
						a_cfg_scale = a_cfg_scale,
						)

				out_current = out[:, self.num_prev_frames:]
				return out_current

			# solve ODE
			trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			sample.append(sample_t)
		sample = torch.cat(sample, dim=1)[:, :T]
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