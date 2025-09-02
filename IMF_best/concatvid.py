import os
import cv2

def concat_videos(root_dir, gen_videos_dirs, out_dir, driving_type="self"):
    """
    root_dir: 包含子目录 (input_frame, driving_xxx.mp4) 的根目录
    gen_videos_dirs: list[str]，多个生成视频目录
    out_dir: 输出目录
    driving_type: "self" 或 "cross"
    """
    os.makedirs(out_dir, exist_ok=True)

    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        # 文件路径
        input_frame_path = os.path.join(sub_path, "input_frame.png")
        driving_video_path = os.path.join(sub_path, f"driving_{driving_type}.mp4")

        # 收集生成视频路径
        gen_video_paths = []
        for gdir in gen_videos_dirs:
            gpath = os.path.join(gdir, f"{sub}.mp4")
            if os.path.exists(gpath):
                gen_video_paths.append(gpath)

        if not (os.path.exists(input_frame_path) and os.path.exists(driving_video_path) and len(gen_video_paths) > 0):
            print(f"skip {sub}, missing file")
            continue

        # 读 input_frame
        input_img = cv2.imread(input_frame_path)

        # 打开 driving 视频
        cap_drive = cv2.VideoCapture(driving_video_path)
        # 打开所有生成视频
        caps_gen = [cv2.VideoCapture(p) for p in gen_video_paths]

        # 确定视频参数
        fps_list = [cap_drive.get(cv2.CAP_PROP_FPS)] + [c.get(cv2.CAP_PROP_FPS) for c in caps_gen]
        fps = min(fps_list)
        width = int(caps_gen[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps_gen[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # resize input_frame 到视频高度
        def resize_to_height(img, h):
            scale = h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, h))

        input_img = resize_to_height(input_img, height)

        # 输出视频writer
        out_path = os.path.join(out_dir, f"{sub}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_width = input_img.shape[1] + width * (1 + len(caps_gen))  # input + driving + 所有生成
        out = cv2.VideoWriter(out_path, fourcc, fps, (total_width, height))

        while True:
            ret_d, frame_d = cap_drive.read()
            if not ret_d:
                break

            frame_d = cv2.resize(frame_d, (width, height))
            frames = [input_img, frame_d]

            valid = True
            gen_frames = []
            for c in caps_gen:
                ret_g, frame_g = c.read()
                if not ret_g:
                    valid = False
                    break
                gen_frames.append(cv2.resize(frame_g, (width, height)))

            if not valid:
                break

            frames.extend(gen_frames)
            concat_frame = cv2.hconcat(frames)
            out.write(concat_frame)

        cap_drive.release()
        for c in caps_gen:
            c.release()
        out.release()
        print(f"saved {out_path}")


if __name__ == "__main__":
    root_dir = r"D:\eval\celebv_video_driven"
    gen_videos_dirs = [
        r"E:\codes\codes\IMF_best\exps\dim32\cross_celebv",
        r"E:\codes\codes\IMF_best\exps\dim32_idtoken_adaptweight\celebv",
        r"E:\codes\codes\IMF_best\exps\dim32_idtoken_adaptweight_triplet\celebv"

    ]
    out_dir = r"E:\codes\codes\IMF_best\exps\dim32_idtoken_adaptweight_triplet\celebv_compare"
    concat_videos(root_dir, gen_videos_dirs, out_dir, driving_type="cross")
