import torch
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from train_512_best import IMFSystem
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import math
def _load_image_cv2(path):
    """使用 OpenCV 加载并转换图像"""
    image = cv2.imread(path)  # BGR
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB
    image = cv2.resize(image, (256, 256))  # 调整尺寸
    image = image.astype(np.float32) / 255.0  # 归一化
    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC → CHW
    return image



def process_single_video(video_frames_dir, model, device, args):
    frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith(('.jpg', '.png'))])
    video_name = os.path.basename(video_frames_dir)
    save_path = os.path.join(args.out_dir, f"{video_name}.pt")
    if os.path.exists(save_path):
        return

    # 预加载图片（并发）
    frame_paths = [os.path.join(video_frames_dir, f) for f in frame_files]
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(_load_image_cv2, frame_paths))

    outputs = []

    # 按 batch 处理
    for i in range(0, len(images), args.batch_size):
        batch_images = images[i:i+args.batch_size]
        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_output = model.encode_latent_token(batch_tensor)

        outputs.extend(batch_output.cpu())

    outputs = torch.stack(outputs, dim=0)
    torch.save(outputs, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help='Directory of input video frames')
    parser.add_argument('--checkpoint', type=str, default='pretrained_models/SMIRK_em1.pt')
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--crop', action='store_true', help='Crop faces based on landmarks')
    parser.add_argument('--debug', action='store_true', help='Save debug images with landmarks')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for processing')
    parser.add_argument("--use_gan", action='store_true', help="Enable GAN training")
    parser.add_argument("--gan_type", type=str, default='lsgan', choices=['lsgan', 'vanilla'])
    parser.add_argument("--gan_weight", type=float, default=0.1, help="Weight for GAN loss")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--display_freq", type=int, default=5000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--exp_path", type=str, default='./exps')
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_l1", type=float, default=1.0)
    parser.add_argument("--loss_vgg", type=float, default=1.0)
    parser.add_argument("--decoder", type=str, default="frame")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="E:\\codes\\codes\\LIA\\a100_rvm\\log\\ckpt\\381000.pt")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--r1_reg_every", type=int, default=16, 
                        help="Frequency to apply R1 regularization (e.g., apply every 16 steps)")
    parser.add_argument("--use_r1_reg", action='store_true', help="gan R1 reg")
    parser.add_argument("--use_2dpos", action='store_true', help="2d position encoding")
    parser.add_argument("--use_arcface", action='store_true', help="use arcface loss")
    parser.add_argument("--arcface_path", type=str, default="E:\codes\codes\model_ir_se50.pth", help="2d position encoding")
    parser.add_argument("--loss_arcface", type=int, default=10)

    # ✨ 新增划分参数
    parser.add_argument('--split', type=int, default=1, help='Total number of splits')
    parser.add_argument('--split_id', type=int, default=0, help='Which split to process (0-indexed)')

    # 以下省略其他参数...

    args = parser.parse_args()

    # ✅ 设置 GPU
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    system = IMFSystem(args)
    if args.model_path:
        system.load_ckpt(args.model_path)
        print(f"Resumed from checkpoint: {args.model_path}")
    model = system.gen
    model.eval()
    model = model.to(device)

    # ✅ 获取所有视频目录
    video_dirs = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, f))]
    video_dirs.sort()  # Optional: 保证顺序一致

    # ✅ 计算当前子区间的数据
    total = len(video_dirs)
    split = args.split
    split_id = args.split_id

    assert 0 <= split_id < split, f"Invalid split_id={split_id} for split={split}"

    per_split = math.ceil(total / split)
    start_idx = split_id * per_split
    end_idx = min((split_id + 1) * per_split, total)

    # ✅ 处理分配的视频子集
    for video_frames_dir in tqdm(video_dirs[start_idx:end_idx], desc=f"Processing split {split_id}/{split} on GPU {args.gpu_id}"):
        video_name = os.path.basename(video_frames_dir)
        try:
            process_single_video(video_frames_dir, model, device, args)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")

if __name__ == '__main__':
    main()

