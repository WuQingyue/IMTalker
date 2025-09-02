import os
import shutil
import subprocess
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip

def extract_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(str(video_path))
    success, frame = cap.read()
    if success:
        cv2.imwrite(str(output_path), frame)
    cap.release()

def extract_audio(video_path, output_path):
    try:
        clip = VideoFileClip(str(video_path))
        clip.audio.write_audiofile(str(output_path), fps=16000, verbose=False, logger=None)
        clip.close()
    except Exception as e:
        print(f"[Audio Error] {video_path.name}: {e}")

def trim_video_ffmpeg(input_path, output_path, duration=10):
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",
            str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FFmpeg Trim Error] {input_path.name}: {e}")
        return False

def process_all_videos(video_dir, output_root_dir):
    video_dir = Path(video_dir)
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {".mp4", ".mov", ".avi", ".mkv"}  # 可根据需要扩展
    all_videos = [p for p in video_dir.iterdir() if p.suffix.lower() in supported_exts]

    for video_path in all_videos:
        vid_id = video_path.stem  # 不含扩展名的文件名作为 ID

        # 每个视频一个子目录
        output_dir = output_root_dir / vid_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 剪辑前10秒的视频
        output_video_path = output_dir / video_path.name
        success = trim_video_ffmpeg(video_path, output_video_path, duration=10)
        if not success:
            print(f"[Skipped] {vid_id} due to ffmpeg trim failure")
            continue

        # 提取第一帧
        frame_path = output_dir / "first_frame.jpg"
        extract_first_frame(output_video_path, frame_path)

        # 提取音频
        audio_path = output_dir / "audio.wav"
        extract_audio(output_video_path, audio_path)

        print(f"[Done] {vid_id}")

# ==== 仅修改这里 ====
video_dir = r"E:\data\vox2_test"
output_dir = r"D:\eval\audio_drive\vox2"
# ===================

process_all_videos(video_dir, output_dir)
