
from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from torchvision import transforms

import os
import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
#def create_eye_mouth_mask(
#    landmarks_68: np.ndarray,
#    image_size: int = 512,
#    # For eyes (optional erosion + dilation)
#    eye_erosion_iters: int = 1,
#    eye_dilate_iters: int = 1,
#    # For mouth
#    mouth_dilate_iters: int = 2
#    ) : 
#    """
#    Create binary masks for eyes and mouth based on 68 facial landmarks.
#    - Eyes: We use the typical 68-landmark indices for left/right eyes,
#            fill them with fillConvexPoly. Then optional erosion/dilation.
#    - Mouth: We ONLY use outer lip indices [48..59], ignore the inner ring [60..67].
#             Then we compute a convex hull to ensure a smooth boundary (no inward spikes),
#             fill that hull, and do morphological dilation to expand the region.
#
#    Args:
#        landmarks_68 (np.ndarray): shape (68,2), each row is (x_norm, y_norm) in [0,1].
#        image_size (int): final mask size (width=height=image_size).
#        eye_erosion_iters (int): how many times to erode eye region before dilate.
#        eye_dilate_iters (int): how many times to dilate eye region.
#        mouth_dilate_iters (int): how many times to dilate mouth region.
#                                  (We skip erosion for mouth in this example.)
#    Returns:
#        eye_mask  (np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
#        mouth_mask(np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
#    """
#    # Initialize empty masks
#    eye_mask = np.zeros((image_size, image_size), dtype=np.uint8)
#    mouth_mask = np.zeros((image_size, image_size), dtype=np.uint8)
#
#    # Indices for left/right eye in 68-landmarks
#    left_eye_idx = [36, 37, 38, 39, 40, 41]
#    right_eye_idx = [42, 43, 44, 45, 46, 47]
#
#    # Outer lips only: [48..59], ignoring [60..67] (inner lips)
#    outer_mouth_idx = list(range(48, 60))
#
#    # Convert normalized coords -> pixel coords
#    def to_px_coords(idx_list):
#        return [
#            (int(landmarks_68[i, 0] * image_size),
#             int(landmarks_68[i, 1] * image_size))
#            for i in idx_list
#        ]
#
#    left_eye_pts = to_px_coords(left_eye_idx)
#    right_eye_pts = to_px_coords(right_eye_idx)
#    mouth_pts = to_px_coords(outer_mouth_idx)
#
#    def fill_polygon(mask, pts):
#        pts_array = np.array(pts, dtype=np.int32)
#        cv2.fillConvexPoly(mask, pts_array, 255)
#
#    # Fill left eye / right eye
#    fill_polygon(eye_mask, left_eye_pts)
#    fill_polygon(eye_mask, right_eye_pts)
#
#    # Mouth: use convex hull on outer-lip points => no inward spikes
#    mouth_pts_array = np.array(mouth_pts, dtype=np.int32)
#    mouth_hull = cv2.convexHull(mouth_pts_array)
#    cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)
#
#    # Morphological ops
#    kernel= np.ones((7, 7), dtype=np.uint8)
#
#    # Eye region: optional erosion then dilation
#    if eye_erosion_iters > 0:
#        eye_mask = cv2.erode(eye_mask, kernel, iterations=eye_erosion_iters)
#    if eye_dilate_iters > 0:
#        eye_mask = cv2.dilate(eye_mask, kernel, iterations=eye_dilate_iters)
#
#    # Mouth region: skip erosion, do dilation to expand outward
#    # (use a slightly bigger kernel / iteration for 512 resolution)
#    if mouth_dilate_iters > 0:
#        mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=mouth_dilate_iters)
#
#    # Convert to float32 binary in [0,1], shape (H,W,1)
#    eye_mask = (eye_mask > 0).astype(np.float32)[None, ...]
#    mouth_mask = (mouth_mask > 0).astype(np.float32)[None, ...]
#
#    return eye_mask, mouth_mask
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_scaled_bbox_mask(
    landmarks_68: np.ndarray,
    image_size: int = 512,
    scale_factor: float = 1.2
) -> np.ndarray:
    """
    Creates a rectangular mask from a bounding box scaled from its center.

    1. Finds the tight bounding box around the 68 landmarks.
    2. Scales this bounding box by the given factor from its center.
    3. Creates a filled rectangular mask based on the scaled box.

    Args:
        landmarks_68 (np.ndarray): shape (68,2), normalized (x,y) in [0,1].
        image_size (int): The width and height of the final square mask.
        scale_factor (float): The factor by which to scale the bounding box (e.g., 1.2 for 120%).

    Returns:
        np.ndarray: The rectangular mask, shape (image_size, image_size, 1), float32 in [0,1].
    """
    # 1. 找到紧凑边界框 (像素坐标)
    pixel_coords = (landmarks_68 * image_size).astype(np.int32)
    x_min = np.min(pixel_coords[:, 0])
    y_min = np.min(pixel_coords[:, 1])
    x_max = np.max(pixel_coords[:, 0])
    y_max = np.max(pixel_coords[:, 1])

    # 2. 计算中心点和原始尺寸
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    center_x = x_min + bbox_width / 2
    center_y = y_min + bbox_height / 2

    # 3. 计算放大后的新尺寸
    new_width = bbox_width * scale_factor
    new_height = bbox_height * scale_factor

    # 4. 计算放大后框的新坐标 (从中心扩展)
    new_x_min = center_x - new_width / 2
    new_y_min = center_y - new_height / 2
    new_x_max = center_x + new_width / 2
    new_y_max = center_y + new_height / 2
    
    # 确保坐标在图像边界内，并转换为整数
    final_x_min = int(np.clip(new_x_min, 0, image_size - 1))
    final_y_min = int(np.clip(new_y_min, 0, image_size - 1))
    final_x_max = int(np.clip(new_x_max, 0, image_size - 1))
    final_y_max = int(np.clip(new_y_max, 0, image_size - 1))

    # 5. 创建并绘制蒙版
    # 初始化一个全黑的蒙版
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 在蒙版上绘制一个白色的实心矩形
    cv2.rectangle(
        mask,
        (final_x_min, final_y_min),  # 左上角坐标
        (final_x_max, final_y_max),  # 右下角坐标
        255,                         # 颜色 (白色)
        -1                           # thickness=-1 表示填充矩形
    )

    # 6. 格式化输出
    # 转换为 float32, 范围 [0, 1], 并增加一个通道维度
    final_mask = (mask > 0).astype(np.float32)[..., np.newaxis]
    
    return final_mask, final_mask

class VFHQ_mask_neg(Dataset):
    def __init__(self, split, db_name):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.split = split
        self.db_path = Path("/mnt/buffer/chenbo")
        self.db_name = db_name

        self._init_path_config()
        self.meta_list = self._load_metadata()

    def _init_path_config(self):
        self.transform = {
            256: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]),
            512: transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        }

    def _load_metadata(self):
        video_root = self.db_path / f"{self.db_name}_video_frame"
        lmd_root = self.db_path / f"{self.db_name}_lmd"

        clip_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
        clip_dirs = clip_dirs[:500]  # 训练/验证统一取前500

        meta_list = []
        for clip_path in tqdm(clip_dirs, desc=f'Processing {self.db_name}'):
            lmd_file = lmd_root / f"{clip_path.name}.txt"
            frame_files = sorted([f for f in clip_path.iterdir() if f.suffix.lower() in ['.png', '.jpg']],
                                 key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            frame_count = len(frame_files)

            if frame_count <= 25 or not lmd_file.is_file():
                continue

            meta_list.append({
                'dir': str(clip_path),
                'frames': frame_files,
                'lmd': str(lmd_file)
            })

        return meta_list

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]
        frame_paths = meta['frames']
        lmd_path = meta['lmd']
    
        # 读取 landmarks
        landmarks = self.read_landmark_info(lmd_path, pixel_scale=(512, 512))
    
        # 校准长度
        min_len = min(len(frame_paths), len(landmarks))
        if min_len < 2:
            return self.__getitem__((idx + 1) % len(self.meta_list))
    
        frame_paths = frame_paths[:min_len]
        landmarks = landmarks[:min_len]
    
        # 随机选一帧 f_id0 和 f_id1
        f_id0, f_id1 = np.random.choice(min_len, size=2, replace=False)
        image_0 = Image.open(frame_paths[f_id0]).convert("RGB")
        image_1 = Image.open(frame_paths[f_id1]).convert("RGB")
        mask_eye_0, mask_mouth_0 = create_eye_mouth_mask(landmarks[f_id0], 512, 0, 2, 2)
        mask_eye_1, mask_mouth_1 = create_eye_mouth_mask(landmarks[f_id1], 512, 0, 2, 2)

        # ------------------------------
        # 负样本采样（不同视频的一帧）
        # ------------------------------
        neg_idx = np.random.randint(len(self.meta_list))
        while neg_idx == idx:  # 避免同视频
            neg_idx = np.random.randint(len(self.meta_list))
        neg_meta = self.meta_list[neg_idx]
        neg_frame_paths = neg_meta['frames']
        neg_lmd_path = neg_meta['lmd']

        neg_landmarks = self.read_landmark_info(neg_lmd_path, pixel_scale=(512, 512))
        neg_len = min(len(neg_frame_paths), len(neg_landmarks))
        neg_frame_id = np.random.randint(neg_len)

        neg_image = Image.open(neg_frame_paths[neg_frame_id]).convert("RGB")
        neg_mask_eye, neg_mask_mouth = create_eye_mouth_mask(neg_landmarks[neg_frame_id], 512, 0, 2, 2)

        return {
            "image_0_256": self.transform[256](image_0),
            "image_1_256": self.transform[256](image_1),
            "image_0_512": self.transform[512](image_0),
            "image_1_512": self.transform[512](image_1),
            "mask_eye_0": torch.tensor(mask_eye_0),
            "mask_mouth_0": torch.tensor(mask_mouth_0),
            "mask_eye_1": torch.tensor(mask_eye_1),
            "mask_mouth_1": torch.tensor(mask_mouth_1),
            # 负样本
            "neg_image_256": self.transform[256](neg_image),
            "neg_image_512": self.transform[512](neg_image),
            "neg_mask_eye": torch.tensor(neg_mask_eye),
            "neg_mask_mouth": torch.tensor(neg_mask_mouth)
        }

    def read_landmark_info(self, lmd_path, pixel_scale, landmark_selected_index=None):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for line in lmd_lines:
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:]
            lmd_obj = []
            if landmark_selected_index:
                for idx in landmark_selected_index:
                    x, y = coords[idx].split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)

class VFHQ_mask(Dataset):
    def __init__(self, split, db_name):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.split = split
        self.db_path = Path("/mnt/buffer/chenbo")
        self.db_name = db_name

        self._init_path_config()
        self.meta_list = self._load_metadata()

    def _init_path_config(self):
        self.transform = {
            256: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]),
            512: transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        }

    def _load_metadata(self):
        video_root = self.db_path / f"{self.db_name}_video_frame"
        lmd_root = self.db_path / f"{self.db_name}_lmd"

        clip_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
        if self.split == 'train':
            clip_dirs = clip_dirs[:500]
        else:
            clip_dirs = clip_dirs[:500]

        meta_list = []
        for clip_path in tqdm(clip_dirs, desc=f'Processing {self.db_name}'):
            lmd_file = lmd_root / f"{clip_path.name}.txt"
            frame_files = sorted([f for f in clip_path.iterdir() if f.suffix.lower() in ['.png', '.jpg']],
                                 key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            frame_count = len(frame_files)

            if frame_count <= 25 or not lmd_file.is_file():
                continue

            meta_list.append({
                'dir': str(clip_path),
                'frames': frame_files,
                'lmd': str(lmd_file)
            })

        return meta_list

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]
        frame_paths = meta['frames']
        lmd_path = meta['lmd']
    
        # 实时读取 landmarks（整个文件读一次）
        landmarks = self.read_landmark_info(lmd_path, pixel_scale=(512, 512))
    
        # 校准长度
        min_len = min(len(frame_paths), len(landmarks))
        if min_len < 2:
            return self.__getitem__((idx + 1) % len(self.meta_list))
    
        frame_paths = frame_paths[:min_len]
        landmarks = landmarks[:min_len]
    
        # 随机选一帧 f_id0 和 f_id1
        f_id0, f_id1 = np.random.choice(min_len, size=2, replace=False)
    
        image_0 = Image.open(frame_paths[f_id0]).convert("RGB")
        image_1 = Image.open(frame_paths[f_id1]).convert("RGB")
    
        mask_eye_0, mask_mouth_0 = create_eye_mouth_mask(landmarks[f_id0], 512, 0, 2, 2)
        mask_eye_1, mask_mouth_1 = create_eye_mouth_mask(landmarks[f_id1], 512, 0, 2, 2)
    
        return {
            "image_0_256": self.transform[256](image_0),
            "image_1_256": self.transform[256](image_1),
            "image_0_512": self.transform[512](image_0),
            "image_1_512": self.transform[512](image_1),
            "mask_eye_0": torch.tensor(mask_eye_0),
            "mask_mouth_0": torch.tensor(mask_mouth_0),
            "mask_eye_1": torch.tensor(mask_eye_1),
            "mask_mouth_1": torch.tensor(mask_mouth_1),
        }

    def read_landmark_info(self, lmd_path, pixel_scale, landmark_selected_index=None):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for line in lmd_lines:
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:]
            lmd_obj = []
            if landmark_selected_index:
                for idx in landmark_selected_index:
                    x, y = coords[idx].split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)

class VFHQ_mask_lmd(Dataset):
    def __init__(self, split, db_name):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.split = split
        self.db_path = Path("/mnt/buffer/chenbo")
        self.db_name = db_name

        self._init_path_config()
        self.meta_list = self._parallel_load_metadata()
        self._precache_resources()

    def _init_path_config(self):
        self.transform = {
            256: transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]),
            512: transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])
        }

    def _parallel_load_metadata(self):
        video_root = self.db_path / f"{self.db_name}_video_frame"
        lmd_root = self.db_path / f"{self.db_name}_lmd"

        clip_dirs = sorted([p for p in video_root.iterdir() if p.is_dir()])
        if self.split == 'train':
            clip_dirs = clip_dirs[:500]
        else:
            clip_dirs = clip_dirs[:500]

        def process_clip(clip_path):
            lmd_file = lmd_root / f"{clip_path.name}.txt"
            frame_files = [f for f in clip_path.iterdir() if f.suffix.lower() in ['.png', '.jpg']]
            frame_count = len(frame_files)

            if frame_count <= 25 or not lmd_file.is_file():
                return None

            with open(lmd_file) as f:
                lmd_count = sum(1 for _ in f)
            if lmd_count <= 25 or frame_count <= lmd_count:
                return None

            return {
                'dir': str(clip_path),
                'count': frame_count,
                'lmd': str(lmd_file)
            }

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(process_clip, clip_dirs), total=len(clip_dirs), desc=f'Processing {self.db_name}'))

        return [r for r in results if r is not None]

    def _precache_resources(self):
        def precache(meta):
            meta_dir = Path(meta['dir'])
            files = sorted([f for f in meta_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg']],
                           key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            landmarks = self.read_landmark_info(meta['lmd'], pixel_scale=(512, 512))

            min_len = min(len(files), len(landmarks))
            if min_len < 2:
                return  # 忽略太短的 clip

            meta['sorted'] = files[:min_len]
            meta['landmarks'] = landmarks[:min_len]
            return meta  # ✅ 如果使用 map 返回 meta

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(precache, self.meta_list), total=len(self.meta_list), desc='Precaching resources'))
        self.meta_list = [r for r in results if r is not None]  # 去掉无效 clip

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        meta = self.meta_list[idx]
        frame_paths = meta['sorted']
        all_landmarks = meta['landmarks']

        if len(frame_paths) < 2:
            return self.__getitem__((idx + 1) % len(self.meta_list))

        # 随机选一帧 f_id0
        f_id0 = np.random.randint(len(frame_paths))
        lmd0 = all_landmarks[f_id0]

        # 计算与其他帧landmark距离，选距离最大的帧 f_id1
        dists = []
        for i, lmd in enumerate(all_landmarks):
            if i == f_id0:
                dists.append(-np.inf)
            else:
                dist = np.linalg.norm(lmd - lmd0)
                dists.append(dist)
        f_id1 = int(np.argmax(dists))

        image_0 = Image.open(frame_paths[f_id0]).convert("RGB")
        image_1 = Image.open(frame_paths[f_id1]).convert("RGB")

        mask_eye_0, mask_mouth_0 = create_eye_mouth_mask(all_landmarks[f_id0], 512, 0, 2, 2)
        mask_eye_1, mask_mouth_1 = create_eye_mouth_mask(all_landmarks[f_id1], 512, 0, 2, 2)

        return {
            "image_0_256": self.transform[256](image_0),
            "image_1_256": self.transform[256](image_1),
            "image_0_512": self.transform[512](image_0),
            "image_1_512": self.transform[512](image_1),
            "mask_eye_0": torch.tensor(mask_eye_0),
            "mask_mouth_0": torch.tensor(mask_mouth_0),
            "mask_eye_1": torch.tensor(mask_eye_1),
            "mask_mouth_1": torch.tensor(mask_mouth_1),
        }

    def read_landmark_info(self, lmd_path, pixel_scale, landmark_selected_index=None):
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for line in lmd_lines:
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:]
            lmd_obj = []
            if landmark_selected_index:
                for idx in landmark_selected_index:
                    x, y = coords[idx].split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale[0], int(y)/pixel_scale[1]))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)
import itertools
import bisect
class CombinedDataset(Dataset):
    def __init__(self, datasets, sampling_probs):
        """
        Args:
            datasets (List[Dataset]): 传入任意数量的 PyTorch Dataset
            sampling_probs (List[float]): 每个 Dataset 的采样概率（之和应为 1）
        """
        assert len(datasets) == len(sampling_probs), "datasets 和 sampling_probs 长度不一致"
        total = sum(sampling_probs)
        self.datasets = datasets
        self.sampling_probs = [p / total for p in sampling_probs]  # 归一化采样概率
        self.cumulative_probs = list(itertools.accumulate(self.sampling_probs))  # 用于快速查找

    def __len__(self):
        # 长度可以设置为所有子数据集长度的总和，也可以自定义
        return sum(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        # 随机选择使用哪个数据集
        rand_val = torch.rand(1).item()
        dataset_idx = bisect.bisect_right(self.cumulative_probs, rand_val)

        # 对该数据集进行随机采样
        dataset = self.datasets[dataset_idx]
        sample_idx = torch.randint(0, len(dataset), (1,)).item()
        return dataset[sample_idx]

def build_video_info(frames, cross_video=False):
    video_info = {}
    for key in frames:
        video_id = get_video_id(key)
        if video_id not in video_info.keys():
            video_info[video_id] = []
        video_info[video_id].append(key)
    for video_id in video_info.keys():
        video_info[video_id] = sorted(
            video_info[video_id], key=lambda x:int(x.split('_')[-1])
        )
    video_mapping = {}
    if cross_video:
        video_ids = list(video_info.keys())
        video_ids = sorted(video_ids)
        for idx, video_id in enumerate(video_ids):
            if idx < len(video_ids) - 1:
                video_mapping[video_id] = video_ids[idx+1]
            else:
                video_mapping[video_id] = video_ids[0]
    return video_info, video_mapping


def get_video_id(frame_key):
    if frame_key.split('_')[0] in ['img']:
        video_id = frame_key.split('_')[1]
    else:
        video_id = frame_key.split('_')[0] 
    return video_id

if __name__ == "__main__":
    train_dataset = VFHQ(split='train', device="cpu")
    print(len(train_dataset))
