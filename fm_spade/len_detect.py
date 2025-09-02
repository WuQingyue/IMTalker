import os
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm

def align_and_convert(audio_dir, motion_dir, required_len, tolerance=2, delete=False):
    audio_dir = Path(audio_dir)
    motion_dir = Path(motion_dir)

    motion_paths = sorted(motion_dir.glob("**/*.pt"))  # 处理 .npy motion 特征文件
    total = 0
    converted = 0
    trimmed = 0
    deleted = 0
    skipped = 0

    for motion_path in tqdm(motion_paths, desc="Processing"):
        audio_path = audio_dir / motion_path.with_suffix(".npy").name  # 与 motion 同名的 audio.npy

        total += 1

        if not audio_path.exists():
            if delete:
                motion_path.unlink(missing_ok=True)
            continue

        try:
            motion = np.load(motion_path)
            audio = np.load(audio_path)

            m_len = motion.shape[0]
            a_len = audio.shape[0]

            # case 1: lengths almost match, trim
            if abs(m_len - a_len) <= tolerance:
                min_len = min(m_len, a_len)
                if min_len < required_len:
                    if delete:
                        motion_path.unlink(missing_ok=True)
                        audio_path.unlink(missing_ok=True)
                        deleted += 1
                    continue
                motion = motion[:min_len]
                audio = audio[:min_len]
                np.save(motion_path, motion)
                np.save(audio_path, audio)
                trimmed += 1
                converted += 1

            # case 2: exact match
            elif m_len == a_len:
                if m_len < required_len:
                    if delete:
                        motion_path.unlink(missing_ok=True)
                        audio_path.unlink(missing_ok=True)
                        deleted += 1
                    continue
                converted += 1  # no action needed, but counted

            # case 3: mismatch and one too short
            elif m_len != a_len and (m_len < required_len or a_len < required_len):
                if delete:
                    motion_path.unlink(missing_ok=True)
                    audio_path.unlink(missing_ok=True)
                    deleted += 1

            # case 4: mismatch but both long enough — skip or log
            else:
                skipped += 1

        except Exception as e:
            if delete:
                motion_path.unlink(missing_ok=True)
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
                deleted += 1

    print("\n[Summary]")
    print(f"  Total processed: {total}")
    print(f"  Converted to .npy (including trimmed): {converted}")
    print(f"  Trimmed for tolerance: {trimmed}")
    print(f"  Deleted due to missing or short: {deleted}")
    print(f"  Skipped due to mismatch: {skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--motion_dir", type=str, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--wav2vec_sec", type=float, default=2.0)
    parser.add_argument("--num_prev_frames", type=int, default=10)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    required_len = int(args.fps * args.wav2vec_sec) + args.num_prev_frames
    align_and_convert(args.audio_dir, args.motion_dir, required_len, delete=args.delete)


