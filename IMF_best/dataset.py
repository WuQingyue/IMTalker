#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

from email.mime import image
import os
import json
import torch
import pickle
import random
import numpy as np
from copy import deepcopy


FOCAL_LENGTH = 12.0

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

def create_eye_mouth_mask(
    landmarks_68: np.ndarray,
    image_size: int = 512,
    # For eyes (optional erosion + dilation)
    eye_erosion_iters: int = 1,
    eye_dilate_iters: int = 1,
    # For mouth
    mouth_dilate_iters: int = 2
    ) : 
    """
    Create binary masks for eyes and mouth based on 68 facial landmarks.
    - Eyes: We use the typical 68-landmark indices for left/right eyes,
            fill them with fillConvexPoly. Then optional erosion/dilation.
    - Mouth: We ONLY use outer lip indices [48..59], ignore the inner ring [60..67].
             Then we compute a convex hull to ensure a smooth boundary (no inward spikes),
             fill that hull, and do morphological dilation to expand the region.

    Args:
        landmarks_68 (np.ndarray): shape (68,2), each row is (x_norm, y_norm) in [0,1].
        image_size (int): final mask size (width=height=image_size).
        eye_erosion_iters (int): how many times to erode eye region before dilate.
        eye_dilate_iters (int): how many times to dilate eye region.
        mouth_dilate_iters (int): how many times to dilate mouth region.
                                  (We skip erosion for mouth in this example.)
    Returns:
        eye_mask  (np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
        mouth_mask(np.ndarray): shape (image_size, image_size, 1), float32 in [0,1].
    """
    # Initialize empty masks
    eye_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    mouth_mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Indices for left/right eye in 68-landmarks
    left_eye_idx = [36, 37, 38, 39, 40, 41]
    right_eye_idx = [42, 43, 44, 45, 46, 47]

    # Outer lips only: [48..59], ignoring [60..67] (inner lips)
    outer_mouth_idx = list(range(48, 60))

    # Convert normalized coords -> pixel coords
    def to_px_coords(idx_list):
        return [
            (int(landmarks_68[i, 0] * image_size),
             int(landmarks_68[i, 1] * image_size))
            for i in idx_list
        ]

    left_eye_pts = to_px_coords(left_eye_idx)
    right_eye_pts = to_px_coords(right_eye_idx)
    mouth_pts = to_px_coords(outer_mouth_idx)

    def fill_polygon(mask, pts):
        pts_array = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts_array, 255)

    # Fill left eye / right eye
    fill_polygon(eye_mask, left_eye_pts)
    fill_polygon(eye_mask, right_eye_pts)

    # Mouth: use convex hull on outer-lip points => no inward spikes
    mouth_pts_array = np.array(mouth_pts, dtype=np.int32)
    mouth_hull = cv2.convexHull(mouth_pts_array)
    cv2.fillConvexPoly(mouth_mask, mouth_hull, 255)

    # Morphological ops
    kernel= np.ones((7, 7), dtype=np.uint8)

    # Eye region: optional erosion then dilation
    if eye_erosion_iters > 0:
        eye_mask = cv2.erode(eye_mask, kernel, iterations=eye_erosion_iters)
    if eye_dilate_iters > 0:
        eye_mask = cv2.dilate(eye_mask, kernel, iterations=eye_dilate_iters)

    # Mouth region: skip erosion, do dilation to expand outward
    # (use a slightly bigger kernel / iteration for 512 resolution)
    if mouth_dilate_iters > 0:
        mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=mouth_dilate_iters)

    # Convert to float32 binary in [0,1], shape (H,W,1)
    eye_mask = (eye_mask > 0).astype(np.float32)[..., None]
    mouth_mask = (mouth_mask > 0).astype(np.float32)[..., None]

    return eye_mask, mouth_mask

class VFHQ(Dataset):
    def __init__(self, split, device):
        super().__init__()
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.split = split
        self.device = device
        self.db_path = "/mnt/buffer/chenbo"
        self.db_list = ["vfhq", "hallo3"]

        self.meta_list = []
        for db_name in tqdm(self.db_list, desc='Loading dbs'):
            current_db_meta_list = []
            video_path = os.path.join(self.db_path, db_name+"_video_frame")
            lmd_path = os.path.join(self.db_path, db_name+"_lmd")
            clip_dirs = os.listdir(video_path)
            train_size = 100
            if self.split == 'train':
                clip_dirs = clip_dirs[:train_size]
            else:
                clip_dirs = clip_dirs[train_size:]
            for clip_name in tqdm(clip_dirs, desc='Loading clips'):
                clip_frames_dir = os.path.join(video_path, clip_name)
                clip_landmarks_dir = os.path.join(lmd_path, clip_name+'.txt')
                frame_len = len(os.listdir(clip_frames_dir))
                if not os.path.exists(clip_landmarks_dir):
                    continue
                landmark_len = len(open(clip_landmarks_dir, 'r').readlines())
                if frame_len != landmark_len:
                    continue
                all_lmd_obj = self.read_landmark_info(clip_landmarks_dir, landmark_selected_index=None)
                current_db_meta_list.append({
                    'clip_frames_dir': clip_frames_dir,
                    'clip_name': clip_name,
                    'all_lmd_obj': all_lmd_obj,
                })
            self.meta_list.extend(current_db_meta_list)
        #self.data_map = {video_id: sorted(glob(os.path.join(self.video_path, video_id, "*.png"))) for video_id in self.video_ids}

        # 统一图像转换
        self.transform = {
            256: transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
            512: transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        }

    def __len__(self):
        return len(self.meta_list)

    def __getitem__(self, idx):
        clip_frames_dir = self.meta_list[idx]['clip_frames_dir']
        all_lmd_obj = self.meta_list[idx]['all_lmd_obj']

        frame_files = sorted([f for f in os.listdir(clip_frames_dir) if f.endswith('.jpg') or f.endswith('.png')])
        frame_paths = [os.path.join(clip_frames_dir, f) for f in frame_files]
        # 随机选择两个不同帧
        f_id, t_id = np.random.choice(len(frame_paths), size=2, replace=False)
        f_frame, t_frame = frame_paths[f_id], frame_paths[t_id]
        target_eye_mask, target_mouth_mask = create_eye_mouth_mask(
            all_lmd_obj[t_id], image_size=512,
            eye_erosion_iters=1, eye_dilate_iters=5,
            mouth_dilate_iters=5
        )
        sample = {
            "f_image": self._load_image(f_frame, 256).to(self.device),
            "t_image": self._load_image(t_frame, 512).to(self.device),
            'target_eye_mask': torch.tensor(target_eye_mask).to(self.device),
            'target_mouth_mask': torch.tensor(target_mouth_mask).to(self.device)
        }

        return sample

    def _load_image(self, path, size):
        """加载图像并转换为所需尺寸"""
        return self.transform[size](Image.open(path).convert("RGB"))

    def read_landmark_info(self, lmd_path, landmark_selected_index=None, pixel_scale=512):
        # print('landmark_selected_index', landmark_selected_index)
        with open(lmd_path, 'r') as file:
            lmd_lines = file.readlines()
        lmd_lines.sort()

        total_lmd_obj = []
        for i, line in enumerate(lmd_lines):
            # Split the coordinates and filter out any empty strings
            coords = [c for c in line.strip().split(' ') if c]
            coords = coords[1:] # do not include the file name in the first row
            lmd_obj = []
            if landmark_selected_index is not None and len(landmark_selected_index) > 0:
                # Ensure that the coordinates are parsed as integers
                for idx in landmark_selected_index:
                    coord_pair = coords[idx]
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale, int(y)/pixel_scale))
            else:
                for coord_pair in coords:
                    x, y = coord_pair.split('_')
                    lmd_obj.append((int(x)/pixel_scale, int(y)/pixel_scale))
            total_lmd_obj.append(lmd_obj)

        return np.array(total_lmd_obj, dtype=np.float32)

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
class Settingtest(Dataset):
    def __init__(self, test_dir, device, setting="self"):
        """
        Args:
            test_dir (str): 数据目录
            device (torch.device): 设备
            setting (str): "self" 或 "cross"，决定加载 self_driving 还是 cross_driving
        """
        super().__init__()
        assert setting in ["self", "cross"], "setting 必须是 'self' 或 'cross'"
        self.test_dir = test_dir
        self.device = device
        self.setting = setting

        # 获取所有测试子目录
        self.video_ids = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        self.data_map = self._prepare_data()

        # 图像转换
        self.transform_256 = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])
        self.transform_512 = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(),
        ])

    def _prepare_data(self):
        """构建子目录到图像和视频的映射"""
        data_map = {}
        for video_id in self.video_ids:
            sub_dir = os.path.join(self.test_dir, video_id)

            # 输入帧
            image_path = next((os.path.join(sub_dir, f) 
                               for f in os.listdir(sub_dir) if f.endswith(('.jpg', '.png'))), None)

            # 驱动视频路径（根据 setting 选择）
            if self.setting == "self":
                video_path = os.path.join(sub_dir, "driving_self.mp4")
            else:
                video_path = os.path.join(sub_dir, "driving_cross.mp4")

            if not image_path or not os.path.exists(video_path):
                continue  # 跳过不完整的数据

            data_map[video_id] = {
                "image": image_path,
                "video": video_path
            }
        return data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_data = self.data_map[video_id]

        # 读取图像
        image_256 = self._load_image(video_data["image"], self.transform_256).to(self.device)
        image_512 = self._load_image(video_data["image"], self.transform_512).to(self.device)

        # 读取视频的每一帧
        video_frames_256, video_frames_512 = self._extract_video_frames(video_data["video"])

        sample = {
            "image_256": image_256,
            "image_512": image_512,
            "video_frames_256": video_frames_256,
            "video_frames_512": video_frames_512,
            "video_id": video_id,
            "setting": self.setting
        }
        return sample

    def _load_image(self, path, transform):
        """加载并转换图像"""
        image = Image.open(path).convert("RGB")
        return transform(image)

    def _extract_video_frames(self, video_path):
        """读取视频的所有帧并转换格式"""
        cap = cv2.VideoCapture(video_path)
        frames_256, frames_512 = [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame)
            frames_256.append(self.transform_256(pil_frame).to(self.device))
            frames_512.append(self.transform_512(pil_frame).to(self.device))
        cap.release()
        return frames_256, frames_512
class VFHQ_test(Dataset):
    def __init__(self, test_dir, device):
        super().__init__()
        self.test_dir = test_dir
        self.device = device

        # 获取所有测试子目录
        self.video_ids = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        self.data_map = self._prepare_data()

        # 图像转换
        self.transform_256 = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

        self.transform_512 = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(),
        ])

    def _prepare_data(self):
        """构建子目录到图像和视频的映射"""
        data_map = {}
        for video_id in self.video_ids:
            sub_dir = os.path.join(self.test_dir, video_id)
            
            # 获取图片和视频文件
            image_path = next((os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(('.jpg', '.png'))), None)
            video_path = next((os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(('.mp4', '.avi', '.mov'))), None)

            if not image_path or not video_path:
                continue  # 跳过不完整的数据

            data_map[video_id] = {
                "image": image_path,
                "video": video_path
            }
        return data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_data = self.data_map[video_id]

        # 读取图像
        image_256 = self._load_image(video_data["image"], self.transform_256).to(self.device)
        image_512 = self._load_image(video_data["image"], self.transform_512).to(self.device)

        # 读取视频的每一帧
        video_frames_256, video_frames_512 = self._extract_video_frames(video_data["video"])

        sample = {
            "image_256": image_256,
            "image_512": image_512,
            "video_frames_256": video_frames_256,
            "video_frames_512": video_frames_512,
            "video_id": video_id,
        }
        return sample

    def _load_image(self, path, transform):
        """加载并转换图像"""
        image = Image.open(path).convert("RGB")
        return transform(image)

    def _extract_video_frames(self, video_path):
        """读取视频的所有帧并转换格式"""
        cap = cv2.VideoCapture(video_path)
        frames_256 = []
        frames_512 = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 读取完毕

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色格式
            pil_frame = Image.fromarray(frame)

            frames_256.append(self.transform_256(pil_frame).to(self.device))
            frames_512.append(self.transform_512(pil_frame).to(self.device))

        cap.release()
        return frames_256, frames_512

class gghead_test(Dataset):
    def __init__(self, test_dir, device):
        super().__init__()
        self.test_dir = test_dir
        self.device = device
        print(test_dir)
        # 获取所有测试子目录
        self.video_ids = sorted([d for d in os.listdir(test_dir)])
        self.data_map = self._prepare_data()

        # 图像转换
        self.transform_256 = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

        self.transform_512 = transforms.Compose([
            transforms.Resize((512, 512)),  
            transforms.ToTensor(),
        ])

    def _prepare_data(self):
        """构建子目录到图像和视频的映射"""
        data_map = {}
        for video_id in self.video_ids:
            video_path = os.path.join(self.test_dir, video_id)
            
            # 获取图片和视频文件
            #image_path = next((os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(('.jpg', '.png'))), None)
            #video_path = next((os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith(('.mp4', '.avi', '.mov'))), None)

            #if not image_path or not video_path:
            #    continue  # 跳过不完整的数据

            data_map[video_id] = {
                #"image": image_path,
                "video": video_path
            }
        return data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_data = self.data_map[video_id]

        # 读取图像
        #image_256 = self._load_image(video_data["image"], self.transform_256).to(self.device)
        #image_512 = self._load_image(video_data["image"], self.transform_512).to(self.device)

        # 读取视频的每一帧
        video_frames_256, video_frames_512 = self._extract_video_frames(video_data["video"])

        sample = {
            #"image_256": image_256,
            #"image_512": image_512,
            "video_frames_256": video_frames_256,
            "video_frames_512": video_frames_512,
            "video_id": video_id,
        }
        return sample

    def _load_image(self, path, transform):
        """加载并转换图像"""
        image = Image.open(path).convert("RGB")
        return transform(image)

    def _extract_video_frames(self, video_path):
        """读取视频的所有帧并转换格式"""
        cap = cv2.VideoCapture(video_path)
        frames_256 = []
        frames_512 = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 读取完毕

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色格式
            pil_frame = Image.fromarray(frame)

            frames_256.append(self.transform_256(pil_frame).to(self.device))
            frames_512.append(self.transform_512(pil_frame).to(self.device))

        cap.release()
        return frames_256, frames_512

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
    train_dataset = VFHQ(split='train', device="cpu")[10]