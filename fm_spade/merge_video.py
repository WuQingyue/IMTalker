import os
import subprocess
import glob

# 定义任意数量的视频路径
input_paths = [
    "D:\eval\celebv",
    r"E:\codes\baseline\hallo\celebv",
    r"E:\codes\baseline\echomimic\celebv_new",
    r"E:\codes\baseline\Sonic\eval_celebv",
    "E:\codes\codes\\fm_spade\exps\\32dim\last-v2_a2_celebv_nullgaze_nullpose_nullcam"
]

output_folder = "E:\codes\codes\\fm_spade\exps\\32dim\last-v2_a2_celebv_nullgaze_nullpose_nullcam_comparediffusion"
os.makedirs(output_folder, exist_ok=True)

# 获取第一个路径中的文件夹名称作为视频名
files = sorted(os.listdir(input_paths[0]))
print(f"共发现 {len(files)} 个视频：{files}")

for file in files:
    # 获取每个路径下的视频文件
    input_files = []
    for i, path in enumerate(input_paths):
#        if i == 0:
#            matches = glob.glob(os.path.join(path, file, "*.mp4"))
#            if not matches:
#                print(f"⚠️ 没找到：{path}\\{file}\\*.mp4")
#                input_files = []
#                break
#            input_files.append(matches[0])
#        else:
        mp4_file = os.path.join(path, file)
        if not os.path.exists(mp4_file):
            print(f"⚠️ 没找到：{mp4_file}")
            input_files = []
            break
        input_files.append(mp4_file)

    if not input_files:
        continue

    # 获取所有输入视频的最短时长
    durations = []
    for f in input_files:
        try:
            duration = float(subprocess.check_output([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', f
            ]))
            durations.append(duration)
        except subprocess.CalledProcessError:
            print(f"⚠️ 获取时长失败：{f}")
            durations = []
            break

    if not durations:
        continue

    min_duration = min(durations)
    print(f"✅ 正在处理 {file}，裁剪至 {min_duration:.2f} 秒")

    # 生成临时裁剪并缩放后的视频
    temp_files = [f"temp_{i}.mp4" for i in range(len(input_files))]
    for inp, temp in zip(input_files, temp_files):
        subprocess.run([
            'ffmpeg', '-i', inp, '-t', str(min_duration),
            '-vf', 'scale=512:512',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-strict', 'experimental', '-y', temp
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 构造 hstack 的 filter 参数
    filter_input = ''.join([f'[{i}:v]' for i in range(len(temp_files))])
    filter_complex = f'{filter_input}hstack=inputs={len(temp_files)}[v]'

    # 构造 ffmpeg 合并命令，添加原始音频作为最后一个输入
    cmd = ['ffmpeg']
    for temp in temp_files:
        cmd.extend(['-i', temp])
    cmd.extend(['-i', input_files[-1]])  # 最后一个视频的原始音频

    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[v]',                      # 使用合成视频轨道
        '-map', f'{len(temp_files)}:a',     # 使用最后一个输入的音频轨道
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-crf', '23',
        '-preset', 'fast',
        '-shortest',                        # 音视频同步，按短的截断
        os.path.join(output_folder, file + ".mp4")
    ])

    subprocess.run(cmd)
    
    # 删除临时文件
    for temp in temp_files:
        os.remove(temp)

print("✅ 所有视频处理完毕！")




