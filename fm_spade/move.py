import os
import shutil
import subprocess

input_root = 'D:\eval\self_reenactment\celebv'      # 包含 video_01, video_02,... 的根目录
output_root = 'D:\eval\\audio_drive\celebv'    # 要保存音频和图片的目标目录

os.makedirs(output_root, exist_ok=True)

for subdir in sorted(os.listdir(input_root)):
    full_path = os.path.join(input_root, subdir)
    if not os.path.isdir(full_path):
        continue

    # 新建对应输出目录
    out_subdir = os.path.join(output_root, subdir)
    os.makedirs(out_subdir, exist_ok=True)

    video_file = None
    image_file = None

    # 遍历子目录寻找视频和图片
    for file in os.listdir(full_path):
        if file.lower().endswith(('.mp4', '.mov', '.mkv')):
            video_file = os.path.join(full_path, file)
        elif file.lower().endswith(('.jpg', '.png')):
            image_file = os.path.join(full_path, file)

    # 提取音频并保存
    if video_file:
        audio_path = os.path.join(out_subdir, 'audio.wav')
        subprocess.run([
            'ffmpeg', '-y', '-i', video_file, '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2', audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 拷贝图片
    if image_file:
        shutil.copy(image_file, os.path.join(out_subdir, os.path.basename(image_file)))

print("处理完成 ✅")
