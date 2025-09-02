from ast import arg
import os
import json
import torch
import numpy as np
from pathlib import Path
import cv2
from multiprocessing import Pool
from tqdm import tqdm

from dataset.audio_processor import AudioProcessor
import argparse


# 初始化 AudioProcessor（按你的参数修改）


# 处理单个音频文件
def process_single_file(audio_path, args):
    audio_name = audio_path.stem
    video_path = Path(args.video_dir) / (audio_name + ".mp4")
    if not video_path.exists():
        return f"[WARN] Video not found: {video_path}"
    cap = cv2.VideoCapture(str(video_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if num_frames <= 0:
        return f"[WARN] Invalid frame count in {video_path}"
    audio_emb = audio_processor.get_embedding(str(audio_path), num_frames)
    save_path = Path(args.out_dir) / (audio_name + ".npy")
    np.save(save_path, audio_emb.cpu().numpy())
    return None  # 成功

# 主函数：顺序处理每一个文件
def main(args):
    audio_paths = sorted(list(Path(args.wav_dir).glob("*.wav")))
    print(f"[INFO] Found {len(audio_paths)} audio files.")

    errors = []
    for audio_path in tqdm(audio_paths):
        result = process_single_file(audio_path, args)
        if result:
            errors.append(result)

    if errors:
        with open("errors.log", "w") as f:
            for line in errors:
                f.write(line + "\n")
        print(f"[INFO] Finished with {len(errors)} errors. See errors.log.")
    else:
        print("[INFO] All done successfully.")

if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav2vec_model_path", type=str, default="E:\codes\codes\FM\wav2vec2-base-960h")
    parser.add_argument("--wav2vec_save_folder", type=str, default='E:\\data\\hdtf\\processed\motion')
    parser.add_argument("--wav_dir", type=str, default='E:\\data\\hdtf\\processed\motion')
    parser.add_argument("--video_dir", type=str, default='E:\\data\\hdtf\\processed\motion')
    parser.add_argument("--out_dir", type=str, default='E:\\data\\hdtf\\processed\motion')
    args = parser.parse_args()

    audio_processor = AudioProcessor(
    sample_rate=16000,
    fps=25,
    wav2vec_model_path=args.wav2vec_model_path,
    only_last_features=True,
    device="cuda"
)
    # demo
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    main(args)
