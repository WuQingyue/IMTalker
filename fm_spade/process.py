from tqdm import tqdm
from models.networks.model_new import IMFModel
import torch.nn as nn
import torch
import os
import random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from torch.utils import data
import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import numpy as np
from dataset.audio_processor import AudioProcessor
import argparse
class hdtf(Dataset):
    def __init__(self, video_dir, wav_dir, device):
        super().__init__()
        self.video_dir = video_dir
        self.wav_dir = wav_dir
        self.device = device

        # 获取所有测试子目录
        self.video_ids = sorted([os.path.splitext(d)[0] for d in os.listdir(video_dir) if d.endswith('.mp4')])
        self.data_map = self._prepare_data()
        # 图像转换
        self.transform_256 = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

    def _prepare_data(self):
        """构建子目录到图像和视频的映射"""
        data_map = []
        for video_id in self.video_ids:
            video_path = os.path.join(self.video_dir, video_id+'.mp4')
            audio_path = os.path.join(self.wav_dir, video_id+'.wav')


            data_map.append({
                "video": video_path,
                "audio": audio_path,
                "video_id": video_id
            })
        return data_map

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        video_data = self.data_map[idx]

        # 读取视频的每一帧
        video_frames_256 = self._extract_video_frames(video_data["video"])

        sample = {
            "video_frames_256": video_frames_256,
            "audio_path": video_data["audio"],
            "video_id": video_data["video_id"],
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 读取完毕

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色格式
            pil_frame = Image.fromarray(frame)

            frames_256.append(self.transform_256(pil_frame))

        cap.release()
        return frames_256

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()
        self.args = args
        self.motion_save_path = args.motion_save_folder
        self.wav2vec_save_path = args.wav2vec_save_folder
        self.model_path = args.model_path
        self.wav2vec_model_path = args.wav2vec_model_path
        self.sample_rate =  16000
        self.fps = 25
        os.makedirs(self.motion_save_path, exist_ok=True)
        os.makedirs(self.wav2vec_save_path, exist_ok=True)
        self.dataset_test = hdtf(video_dir="E:\\data\\hdtf\\video_25fps", wav_dir="E:\\data\\hdtf\\audio_16k",  device="cuda")
        self.loader_test = data.DataLoader(
            self.dataset_test,
            num_workers=0,
            batch_size=1,
            sampler=None,
            pin_memory=False,
            drop_last=False,
        )
        self.gen = IMFModel().to("cuda")
        weight = torch.load(self.model_path, map_location=lambda storage, loc: storage )['gen']
        self.gen.load_state_dict(weight,strict=False)
        self.gen.eval()
		#self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

		# wav2vec2 audio preprocessor
        self.audio_processor = AudioProcessor(self.sample_rate,
                                         self.fps,
                                         self.wav2vec_model_path,
                                         only_last_features = True,
                                         device="cuda")

    def run(self):
        print('==> running')
        with torch.no_grad():
            pbar = tqdm(range(len(self.dataset_test)), desc="Inferencing Progress")
            loader = sample_data(self.loader_test)
            for idx in pbar:
                batch = next(loader)
                video_frames = batch["video_frames_256"]
                audio_path = batch["audio_path"][0]
                video_id = batch["video_id"][0]

                wav2vec_save_path = os.path.join(self.wav2vec_save_path, video_id + ".npy")
                motion_save_path = os.path.join(self.motion_save_path, video_id + ".npy")
                audio_emb, audio_length = self.audio_processor.get_embedding(audio_path, len(video_frames))
                np.save(wav2vec_save_path, audio_emb)

#                t_list = []
#                for i in tqdm(range(len(video_frames))):
#                    t_c = self.gen.encode_latent_token(video_frames[i].cuda())
#                    t_list.append(t_c.cpu().numpy())
#
#                    # optional: delete intermediate tensor and clear cache
#                    del t_c
#                    torch.cuda.empty_cache()
#
#                token = np.array(t_list)
#                print(token.shape, audio_emb.shape)
               #np.save(wav2vec_save_path, audio_emb)
#                np.save(motion_save_path, token)



if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec_model_path", type=str, default="E:\codes\codes\FM\wav2vec2-base-960h")
    parser.add_argument("--motion_save_folder", type=str, default='E:\\data\\hdtf\\processed\wav2vec2')
    parser.add_argument("--wav2vec_save_folder", type=str, default='E:\\data\\hdtf\\processed\motion')
    parser.add_argument("--model_path", type=str, default="E:\codes\codes\IMF_lambda\exps\\10l1_10vgg_1gan_batch16_lr1e-4\checkpoint\\240000.pt")
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    demo.run()