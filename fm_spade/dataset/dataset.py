from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import torch
import random
from options.base_options import BaseOptions
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
def load_pose(smirk):
    pose = smirk["pose_params"]  # (1, 3)
    cam = smirk["cam"]              # (1, 3)
    return pose, cam
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random
from tqdm import tqdm
import json
class AudioMotionSmirkGazeDataset(Dataset):
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        with open("dataset_meta.json", "r") as f:
            all_meta = json.load(f)

        self.samples = []
        for item in tqdm(all_meta[start:end], desc="Filtering valid samples"):
            if min(item["motion_len"], item["audio_len"], item.get("gaze_len", 1e9), item.get("smirk_len", 1e9)) >= self.required_len:
                self.samples.append({
                    "motion_path": item["motion_path"],
                    "audio_path": item["audio_path"],
                    "smirk_path": item["smirk_path"],  # 新增
                    "gaze_path":  item["gaze_path"]    # 新增
                })

        if not self.samples:
            raise RuntimeError("No valid samples found.")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def _get_full_clip(self, index):
        """加载 motion/audio/pose/gaze 的完整 clip 并切分 prev/now。"""
        item = self.samples[index]
        motion = torch.load(item['motion_path'])
        audio = np.load(item['audio_path'], mmap_mode='r')
        gaze = np.load(item['gaze_path'], mmap_mode='r')
        smirk = torch.load(item['smirk_path'])
        pose, cam = load_pose(smirk)

        min_len = min(len(motion), len(audio), len(gaze), len(pose))
        start_idx = random.randint(0, min_len - self.required_len)

        audio_seg = torch.from_numpy(audio[start_idx:start_idx + self.required_len].copy()).float()
        motion_seg = motion[start_idx:start_idx + self.required_len]
        gaze_seg = torch.from_numpy(gaze[start_idx:start_idx + self.required_len].copy()).float()
        pose_seg = pose[start_idx:start_idx + self.required_len]
        cam_seg = cam[start_idx:start_idx + self.required_len]

        motion_prev = motion_seg[:self.num_prev_frames]
        motion_clip = motion_seg[self.num_prev_frames:]
        audio_prev = audio_seg[:self.num_prev_frames]
        audio_clip = audio_seg[self.num_prev_frames:]
        gaze_prev = gaze_seg[:self.num_prev_frames]
        gaze_clip = gaze_seg[self.num_prev_frames:]
        pose_prev = pose_seg[:self.num_prev_frames]
        pose_clip = pose_seg[self.num_prev_frames:]
        cam_prev = cam_seg[:self.num_prev_frames]
        cam_clip = cam_seg[self.num_prev_frames:]

        return motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev

    def __getitem__(self, index):
        try:
            motion_clip, audio_clip, motion_prev, audio_prev, motion_seg, gaze_clip, gaze_prev, pose_clip, pose_prev, cam_clip, cam_prev = self._get_full_clip(index)
        except Exception as e:
            print(f"[Error] Failed to get clip for index {index}: {e}. Trying a random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        ref_idx = torch.randint(low=0, high=motion_seg.shape[0], size=(1,)).item()
        m_ref = motion_seg[ref_idx]

        return {
            "m_now": motion_clip,
            "a_now": audio_clip,
            "gaze_now": gaze_clip,
            "pose_now": pose_clip,
            "cam_now": cam_clip, 
            "m_prev": motion_prev,
            "a_prev": audio_prev,
            "gaze_prev": gaze_prev,
            "pose_prev": pose_prev,
            "cam_prev": cam_prev,
            "m_ref": m_ref,
        }
class AudioMotionDataset_cl(Dataset):
    """
    该数据集为对比学习准备数据。
    对于每个样本，它提供：
    - 一个待去噪的目标运动序列 (m_now) 和其上下文 (audio_clip, etc.)
    - m_ref_anchor: 从 m_now 中随机抽取的一帧作为锚点参考。
    - m_ref_pos: 从 m_now 中随机抽取的另一帧作为正样本参考。
    - m_ref_negs: 从多个完全不同的视频中随机抽取的N帧作为负样本参考。
    """
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.audio_folder = Path(self.opt.audio_path)
        self.motion_folder = Path(self.opt.motion_path)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames
        
        # 新增：可配置的负样本数量，如果opt中未定义，则默认为1以保持向后兼容
        self.num_neg_samples = getattr(self.opt, 'num_neg_samples', 1)

        if self.num_frames_for_clip < 2:
            raise ValueError("self.num_frames_for_clip 必须至少为 2，才能采样 anchor 和 positive 参考帧。")

        self.samples = []
        all_motion_paths = sorted(self.motion_folder.glob("**/*.pt"))[start:end]
        print(f"[Info] Found {len(all_motion_paths)} motion files. Verifying...")

        # ... (数据预处理部分保持不变)
        for motion_path in tqdm(all_motion_paths, desc="Preprocessing Data"):
            audio_path = self.audio_folder / motion_path.with_suffix(".npy").name
            if not audio_path.exists():
                continue
            try:
                motion_len = torch.load(motion_path).shape[0]
                audio_len = np.load(audio_path, mmap_mode='r').shape[0]
                if min(motion_len, audio_len) >= self.required_len:
                    self.samples.append({
                        'audio_path': str(audio_path),
                        'motion_path': str(motion_path),
                    })
            except Exception as e:
                print(f"[Warn] Skipping file {motion_path.name} due to error: {e}")
        
        if not self.samples:
            raise RuntimeError("No valid samples found.")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

        # 新增：确保数据集足够大以提供指定数量的唯一负样本
        if len(self.samples) <= self.num_neg_samples:
            raise ValueError(f"数据集样本总数 ({len(self.samples)}) 必须大于请求的负样本数量 ({self.num_neg_samples})。")

    def __len__(self):
        return len(self.samples)

    def _get_full_clip(self, index):
        """辅助函数，加载并返回一个完整的随机片段及其上下文。(保持不变)"""
        item = self.samples[index]
        audio_path, motion_path = item['audio_path'], item['motion_path']
        
        motion = torch.load(motion_path)
        audio = np.load(audio_path, mmap_mode='r')
        min_len = min(len(motion), len(audio))
        
        start_idx = random.randint(0, min_len - self.required_len)
        
        audio_seg = torch.from_numpy(audio[start_idx : start_idx + self.required_len].copy()).float()
        motion_seg = motion[start_idx : start_idx + self.required_len]

        motion_prev = motion_seg[:self.num_prev_frames]
        motion_clip = motion_seg[self.num_prev_frames:]
        audio_prev = audio_seg[:self.num_prev_frames]
        audio_clip = audio_seg[self.num_prev_frames:]
        
        return motion_clip, audio_clip, motion_prev, audio_prev
        
    def __getitem__(self, index):
        # 1. 获取目标序列 (anchor/positive 的来源) (逻辑不变)
        try:
            motion_clip, audio_clip, motion_prev, audio_prev = self._get_full_clip(index)
        except Exception as e:
            print(f"[Error] Failed to get clip for index {index}: {e}. Trying a random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 2. 从目标序列中采样 anchor 和 positive 参考帧 (逻辑不变)
        ref_indices = torch.randperm(motion_clip.shape[0])[:2]
        m_ref_anchor = motion_clip[ref_indices[0]]
        m_ref_pos = motion_clip[ref_indices[1]]
        
        # 3. 修改：获取 N 个负样本参考帧
        # 创建一个不包含当前样本索引的索引池
        possible_indices = list(range(len(self)))
        possible_indices.remove(index)
        
        # 从索引池中随机、不重复地抽取N个索引
        neg_indices = random.sample(possible_indices, self.num_neg_samples)
        
        m_ref_negs_list = []
        for neg_idx in neg_indices:
            try:
                motion_clip_neg, _, _, _ = self._get_full_clip(neg_idx)
                neg_frame = motion_clip_neg[random.randint(0, motion_clip_neg.shape[0] - 1)]
                m_ref_negs_list.append(neg_frame)
            except Exception:
                # 如果某个负样本出错，可以简单地跳过或用另一个随机样本替换
                # 这里我们选择用另一个随机样本替换，确保数量不变
                fallback_idx = random.choice(possible_indices)
                motion_clip_neg, _, _, _ = self._get_full_clip(fallback_idx)
                neg_frame = motion_clip_neg[random.randint(0, motion_clip_neg.shape[0] - 1)]
                m_ref_negs_list.append(neg_frame)

        # 将负样本列表堆叠成一个张量
        m_ref_negs = torch.stack(m_ref_negs_list, dim=0)

        return {
            "m_now": motion_clip,
            "a_now": audio_clip,
            "m_prev": motion_prev,
            "a_prev": audio_prev,
            "m_ref_anchor": m_ref_anchor,
            "m_ref_pos": m_ref_pos,
            "m_ref_negs": m_ref_negs, # <-- 返回包含N个负样本的张量
        }

class AudioMotionDataset(Dataset):
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames
        self.num_neg_samples = getattr(self.opt, 'num_neg_samples', 1)

        with open("dataset_meta.json", "r") as f:
            all_meta = json.load(f)

        self.samples = []
        for item in tqdm(all_meta[start:end], desc="Filtering valid samples"):
            if min(item["motion_len"], item["audio_len"]) >= self.required_len:
                self.samples.append({
                    "motion_path": item["motion_path"],
                    "audio_path": item["audio_path"]
                })

        if not self.samples:
            raise RuntimeError("No valid samples found.")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

        # 新增：确保数据集足够大以提供指定数量的唯一负样本
    def __len__(self):
        return len(self.samples)

    def _get_full_clip(self, index):
        """辅助函数，加载并返回一个完整的随机片段及其上下文。(保持不变)"""
        item = self.samples[index]
        audio_path, motion_path = item['audio_path'], item['motion_path']
        
        motion = torch.load(motion_path)
        audio = np.load(audio_path, mmap_mode='r')
        min_len = min(len(motion), len(audio))
        
        start_idx = random.randint(0, min_len - self.required_len)
        
        audio_seg = torch.from_numpy(audio[start_idx : start_idx + self.required_len].copy()).float()
        motion_seg = motion[start_idx : start_idx + self.required_len]

        motion_prev = motion_seg[:self.num_prev_frames]
        motion_clip = motion_seg[self.num_prev_frames:]
        audio_prev = audio_seg[:self.num_prev_frames]
        audio_clip = audio_seg[self.num_prev_frames:]
        
        return motion_clip, audio_clip, motion_prev, audio_prev, motion_seg
        
    def __getitem__(self, index):
        # 1. 获取目标序列 (anchor/positive 的来源) (逻辑不变)
        try:
            motion_clip, audio_clip, motion_prev, audio_prev, motion_seg = self._get_full_clip(index)
        except Exception as e:
            print(f"[Error] Failed to get clip for index {index}: {e}. Trying a random sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))
        ref_idx = torch.randint(low=0, high=motion_clip.shape[0], size=(1,)).item()
        m_ref = motion_clip[ref_idx]
        

        return {
            "m_now": motion_clip,
            "a_now": audio_clip,
            "m_prev": motion_prev,
            "a_prev": audio_prev,
            "m_ref": m_ref,
        }

class DynamicAudioMotionDataset(Dataset):
    # 1. 在构造函数中增加 max_seq_len 参数
    def __init__(self, opt,  start=0, end=None):
        super().__init__()
        self.opt = opt
        self.min_prev_len = opt.min_prev_len
        self.min_now_len = opt.min_now_len
        self.max_seq_len = opt.max_seq_len  # <--- 新增：保存最大长度上限
        self.samples = []

        with open("dataset_meta.json", "r") as f:
            all_meta = json.load(f)

        min_required_len = self.min_prev_len + self.min_now_len
        for item in tqdm(all_meta[start:end], desc="Filtering valid samples"):
            min_len = min(item["motion_len"], item["audio_len"])
            # 过滤逻辑保持不变，确保至少满足最小长度要求
            if min_len >= min_required_len:
                self.samples.append({
                    "motion_path": item["motion_path"],
                    "audio_path": item["audio_path"],
                    "min_len": min_len
                })

        if not self.samples:
            raise RuntimeError("No valid samples found.")
        print(f"[Info] Collected {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def _get_dynamic_clip(self, item):
        motion = torch.load(item["motion_path"])
        # Add .copy() after loading the array
        audio = torch.from_numpy(np.load(item["audio_path"], mmap_mode='r').copy()).float()

        max_valid_len = min(len(motion), len(audio))
        min_len_for_clip = self.min_prev_len + self.min_now_len

        # 2. 修改采样长度的逻辑
        #    采样的长度上限现在是「配置的最大长度」和「当前文件有效长度」中的较小者
        upper_bound = min(max_valid_len, self.max_seq_len)

        # 防御性检查：如果该文件的有效长度上限比我们要求的最小长度还小，
        # 就无法采样，直接抛出异常，由 __getitem__ 的 try-except 机制处理
        if upper_bound < min_len_for_clip:
            raise ValueError(f"Clip is too short to sample. Required: {min_len_for_clip}, Available: {upper_bound}")

        # 在新的、有上限的范围内随机选择总长度
        total_len = random.randint(min_len_for_clip, upper_bound)

        start_idx = random.randint(0, max_valid_len - total_len)
        segment_motion = motion[start_idx: start_idx + total_len]
        segment_audio = audio[start_idx: start_idx + total_len]

        split_point = random.randint(self.min_prev_len, total_len - self.min_now_len)
        
        ref_idx_in_now = torch.randint(low=0, high=total_len - split_point, size=(1,)).item()
        m_ref = segment_motion[split_point + ref_idx_in_now]

        return segment_motion, segment_audio, split_point, m_ref

    def __getitem__(self, index):
        try:
            item = self.samples[index]
            segment_motion, segment_audio, split_point, m_ref = self._get_dynamic_clip(item)
            
            return {
                "m_full": segment_motion,
                "a_full": segment_audio,
                "split_point": split_point,
                "m_ref": m_ref
            }
        except Exception as e:
            # 当 _get_dynamic_clip 抛出异常（如片段太短）时，这里会捕获并尝试另一个随机样本
            # print(f"[Warning] Failed index {index}: {e}, trying another...")
            return self.__getitem__(random.randint(0, len(self) - 1))

    @staticmethod
    def collate_fn(batch):
        # collate_fn 无需任何改动
        m_full_padded = pad_sequence([item["m_full"] for item in batch], batch_first=True)
        a_full_padded = pad_sequence([item["a_full"] for item in batch], batch_first=True)
        split_points = torch.tensor([item["split_point"] for item in batch], dtype=torch.long)
        m_refs = torch.stack([item["m_ref"] for item in batch])
        m_full_lens = torch.tensor([item["m_full"].shape[0] for item in batch], dtype=torch.long)
        
        return {
            "m_full": m_full_padded,
            "a_full": a_full_padded,
            "m_ref": m_refs,
            "split_points": split_points,
            "m_full_lens": m_full_lens
        }
class AudioSmirkDataset(Dataset):
    def __init__(self, opt, start, end):
        super().__init__()
        self.opt = opt
        self.audio_folder = Path(self.opt.audio_path)
        self.motion_folder = Path(self.opt.motion_path)
        self.smirk_folder = Path(self.opt.smirk_path)
        self.gaze_folder = Path(self.opt.gaze_path)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        self.samples = []

        motion_paths = sorted(self.motion_folder.glob("**/*.pt"))[start:end]
        print(f"[Info] Found {len(motion_paths)} motion files")

        for motion_path in motion_paths:
            audio_path = self.audio_folder / motion_path.with_suffix(".npy").name
            gaze_path = self.gaze_folder / motion_path.with_suffix(".npy").name
            smirk_path = self.smirk_folder / motion_path.with_suffix(".pt").name
            if audio_path.exists() and gaze_path.exists() and smirk_path.exists():

                self.samples.append({
                    'audio_path': str(audio_path),
                    'motion_path': str(motion_path),
                    'smirk_path': str(smirk_path),
                    'gaze_path': str(gaze_path)
                })

        print(f"[Info] Collected {len(self.samples)} samples (no pre-validation)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        audio_path = item['audio_path']
        motion_path = item['motion_path']
        smirk_path = item['smirk_path']
        gaze_path  = item['gaze_path']

        try:
            audio = np.load(audio_path, mmap_mode='r')
            motion = torch.load(motion_path)
            gaze = np.load(gaze_path, mmap_mode='r')
            smirk = torch.load(smirk_path)
            pose = load_pose(smirk)
            min_len = min(len(audio), len(motion), len(gaze), len(pose))
            
            audio = audio[:min_len]
            motion = motion[:min_len]
            gaze = gaze[:min_len]
            pose = pose[:min_len]
            
            if min_len < self.required_len:
                raise ValueError(f"[Skip] Sample too short: {motion_path} ({min_len} < {self.required_len})")

            start_idx = random.randint(0, min_len - self.required_len)
            audio = torch.from_numpy(audio[start_idx : start_idx + self.required_len].copy()).float()
            motion = motion[start_idx : start_idx + self.required_len]
            gaze = torch.from_numpy(gaze[start_idx : start_idx + self.required_len].copy()).float()
            pose = pose[start_idx : start_idx + self.required_len]

            motion_prev = motion[:self.num_prev_frames]
            motion_clip = motion[self.num_prev_frames:]
            audio_prev = audio[:self.num_prev_frames]
            audio_clip = audio[self.num_prev_frames:]
            pose_prev = pose[:self.num_prev_frames]
            pose_clip = pose[self.num_prev_frames:]
            gaze_prev = gaze[:self.num_prev_frames]
            gaze_clip = gaze[self.num_prev_frames:]

            ref_idx = random.randint(0, motion_clip.shape[0] - 1)
            motion_ref = motion_clip[ref_idx]  # shape: (motion_dim)
            return {
                "m_now": motion_clip,
                "a_now": audio_clip,
                "pose_now": pose_clip,
                "gaze_now": gaze_clip,
                "m_prev": motion_prev,
                "a_prev": audio_prev,
                "pose_prev": pose_prev,
                "gaze_prev": gaze_prev,
                "m_ref": motion_ref
            }

        except Exception as e:
            print(f"[Warn] Skipping index {index} due to error: {e}")

            # 尝试下一个样本（递归），防止无限递归时崩溃
            next_index = (index + 1) % len(self)
            if next_index == index:
                raise RuntimeError("No valid samples in dataset.")
            return self.__getitem__(next_index)



class AudioMotionTestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.audio_folder = Path(self.opt.audio_path)
        self.motion_folder = Path(self.opt.motion_path)
        self.pose_folder = Path(self.opt.pose_path)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        self.samples = []

        motion_paths = sorted(self.motion_folder.glob("**/*.npy"))[-100:]
        print(f"[Info] Found {len(motion_paths)} motion files")

        for motion_path in motion_paths:
            audio_path = self.audio_folder / motion_path.name
            pose_path = self.pose_folder / motion_path.with_suffix(".pt").name
            if audio_path.exists():
                self.samples.append({
                    'audio_path': str(audio_path),
                    'motion_path': str(motion_path),
                    'pose_path': str(pose_path),
                })

        print(f"[Info] Collected {len(self.samples)} samples (no pre-validation)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        audio_path = item['audio_path']
        motion_path = item['motion_path']
        pose_path = item['pose_path']

        try:
            audio = np.load(audio_path, mmap_mode='r')
            motion = np.load(motion_path, mmap_mode='r')
            pose = torch.load(pose_path)
            if len(audio) != len(motion):
                raise ValueError(f"[Skip] Length mismatch: {motion_path} (audio={len(audio)}, motion={len(motion)})")

            total_frames = len(motion)
            if total_frames < self.required_len:
                raise ValueError(f"[Skip] Sample too short: {motion_path} ({total_frames} < {self.required_len})")

            start_idx = random.randint(0, total_frames - self.required_len)

            audio = torch.from_numpy(audio[start_idx : start_idx + self.required_len].copy()).float()
            motion = torch.from_numpy(motion[start_idx : start_idx + self.required_len].copy()).float()
            pose = load_smirk(pose)[start_idx : start_idx + self.required_len]

            motion_prev = motion[:self.num_prev_frames]
            motion_clip = motion[self.num_prev_frames:]
            audio_prev = audio[:self.num_prev_frames]
            audio_clip = audio[self.num_prev_frames:]
            pose_prev = pose[:self.num_prev_frames]
            pose_clip = pose[self.num_prev_frames:]

            return {
                "m_now": motion_clip,
                "a_now": audio_clip,
                "pose_now": pose_clip,
                "m_prev": motion_prev,
                "a_prev": audio_prev,
                "pose_prev": pose_prev,
            }

        except Exception as e:
            print(f"[Warn] Skipping index {index} due to error: {e}")

            # 尝试下一个样本（递归），防止无限递归时崩溃
            next_index = (index + 1) % len(self)
            if next_index == index:
                raise RuntimeError("No valid samples in dataset.")
            return self.__getitem__(next_index)




class AudioMotionPoseTrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.audio_folder = Path(self.opt.audio_path)
        self.motion_folder = Path(self.opt.motion_path)
        self.pose_folder = Path(self.opt.pose_path)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        self.preloaded_data = []
        self.valid_ids = []  # ✅ 存放合法的样本 ID（文件名）

        motion_paths = sorted(list(self.motion_folder.glob("**/*.npy")))
        motion_paths = motion_paths[:-50]
        print(f"[Info] Found {len(motion_paths)} motion files")

        for motion_path in motion_paths:
            audio_path = self.audio_folder / motion_path.name
            pose_path = self.pose_folder / motion_path.with_suffix(".pt").name
            
            if not audio_path.exists():
                continue
            try:
                a = np.load(audio_path)
                t = np.load(motion_path)
                pose = torch.load(pose_path)
                if a.shape[0] == t.shape[0] and t.shape[0] >= self.required_len:
                    self.preloaded_data.append({
                        'a': torch.from_numpy(a).float(),
                        't': torch.from_numpy(t).float(),
                        'pose': pose
                    })
                    
                    self.valid_ids.append(motion_path.stem)  # ✅ 保存合法文件名（不含扩展名）
                else:
                    print(a.shape[0], t.shape[0], motion_path.name)
            except Exception as e:
                print(f"[Warning] Skipped {motion_path.name} due to error: {e}")

        print(f"[Info] Preloaded {len(self.preloaded_data)} samples (from {len(motion_paths)} motion files)")


    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, index):
        data = self.preloaded_data[index]
        audio = data['a']
        motion = data['t']
        pose = data['pose']
        total_frames = motion.shape[0]
        clip_len = self.num_frames_for_clip
        prev_len = self.num_prev_frames
        required_len = clip_len + prev_len

        start_idx = random.randint(0, total_frames - required_len)

        motion_prev = motion[start_idx : start_idx + prev_len]
        motion_clip = motion[start_idx + prev_len : start_idx + required_len]

        audio_prev = audio[start_idx : start_idx + prev_len]
        audio_clip = audio[start_idx + prev_len : start_idx + required_len]

        pose = pose[start_idx: start_idx + required_len]
        return {
            "m_now": motion_clip.squeeze(),
            "a_now": audio_clip,
            "m_prev": motion_prev.squeeze(),
            "a_prev": audio_prev,
            "pose": load_smirk(pose)
        }


class AudioMotionPoseTestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.audio_folder = Path(self.opt.audio_path)
        self.motion_folder = Path(self.opt.motion_path)
        self.pose_folder = Path(self.opt.pose_path)

        self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
        self.num_prev_frames = int(self.opt.num_prev_frames)
        self.required_len = self.num_frames_for_clip + self.num_prev_frames

        self.preloaded_data = []
        self.valid_ids = []  # ✅ 存放合法的样本 ID（文件名）

        motion_paths = sorted(list(self.motion_folder.glob("**/*.npy")))
        motion_paths = motion_paths[-50:]
        print(f"[Info] Found {len(motion_paths)} motion files")

        for motion_path in motion_paths:
            audio_path = self.audio_folder / motion_path.name
            pose_path = self.pose_folder / motion_path.with_suffix(".pt").name
            
            if not audio_path.exists():
                continue
            try:
                a = np.load(audio_path)
                t = np.load(motion_path)
                pose = torch.load(pose_path)
                if a.shape[0] == t.shape[0] and t.shape[0] >= self.required_len:
                    self.preloaded_data.append({
                        'a': torch.from_numpy(a).float(),
                        't': torch.from_numpy(t).float(),
                        'pose': pose
                    })
                    
                    self.valid_ids.append(motion_path.stem)  # ✅ 保存合法文件名（不含扩展名）
                else:
                    print(a.shape[0], t.shape[0], motion_path.name)
            except Exception as e:
                print(f"[Warning] Skipped {motion_path.name} due to error: {e}")

        print(f"[Info] Preloaded {len(self.preloaded_data)} samples (from {len(motion_paths)} motion files)")


    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, index):
        data = self.preloaded_data[index]
        audio = data['a']
        motion = data['t']
        pose = data['pose']
        total_frames = motion.shape[0]
        clip_len = self.num_frames_for_clip
        prev_len = self.num_prev_frames
        required_len = clip_len + prev_len

        start_idx = random.randint(0, total_frames - required_len)

        motion_prev = motion[start_idx : start_idx + prev_len]
        motion_clip = motion[start_idx + prev_len : start_idx + required_len]

        audio_prev = audio[start_idx : start_idx + prev_len]
        audio_clip = audio[start_idx + prev_len : start_idx + required_len]

        pose = pose[start_idx: start_idx + required_len]
        return {
            "m_now": motion_clip.squeeze(),
            "a_now": audio_clip,
            "m_prev": motion_prev.squeeze(),
            "a_prev": audio_prev,
            "pose": load_smirk(pose)
        }
#    def __getitem__(self, index):
#        data = self.preloaded_data[index]
#        audio = data['a']
#        motion = data['t']
#
#        return {
#            "a": audio[:250,...],
#            "ref_x": motion[0].squeeze(),
#            "x": motion[:250,...].squeeze()
#        }

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        super().initialize(parser)
        parser.add_argument("--audio_path",
                            default=None, type=str, help='dataset path')
        parser.add_argument("--motion_path",
                            default=None, type=str, help='dataset path')
        parser.add_argument('--lr',
                            default=1e-4, type=float, help='learning rate')
        parser.add_argument('--batch_size',
                            default=128, type=int, help='batch size')
        parser.add_argument('--iter',
                            default=1000000, type=int, help='training iters')
        parser.add_argument('--res_video_path',
                            default=None, type=str, help='res video path')
        parser.add_argument('--ckpt_path',
                            default="/home/nvadmin/workspace/taek/float-pytorch/checkpoints/float.pth", type=str, help='checkpoint path')
        parser.add_argument('--res_dir',
                            default="./results", type=str, help='result dir')
        parser.add_argument("--exp_path", type=str, default='./exps')
        parser.add_argument("--exp_name", type=str, default='debug')
        parser.add_argument("--save_freq", type=int, default=100000)
        parser.add_argument("--display_freq", type=int, default=10000)
        parser.add_argument("--resume_ckpt", type=str, default=None)
        parser.add_argument("--rank", type=str, default="cuda")
        return parser
    
if __name__ == "__main__":
    opt = TrainOptions().parse()
    dataset = AudioMotionDataset(opt, split="val")

    with open("val_ids.txt", "w") as f:
        for vid in dataset.valid_ids:
            f.write(vid + "\n")



