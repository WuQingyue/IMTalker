import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

motion_dir = Path("E:\codes\codes\IMF_best\exps\\best_2transformer_512dim_spade\exps\\best_2transformer_512dim_spade\\tokens")
audio_dir = Path("E:\data\\voxceleb2\wav2vec2")
smirk_dir = Path()
gaze_dir = Path()

meta = []

for motion_path in tqdm(sorted(motion_dir.glob("**/*.pt"))):
    audio_path = audio_dir / motion_path.with_suffix(".npy").name
    gaze_path = gaze_dir / motion_path.with_suffix(".npy").name
    smirk_path = smirk_dir / motion_path.with_suffix(".pt").name
    if not (audio_path.exists() and gaze_path.exists() and smirk_path.exists()):
        continue
    try:
        motion_len = torch.load(motion_path, map_location='cpu').shape[0]
        audio_len = np.load(audio_path, mmap_mode='r').shape[0]
        gaze = np.load(gaze_path, mmap_mode='r')
        smirk = torch.load(smirk_path)["pose_params"]
        meta.append({
            "motion_path": str(motion_path),
            "audio_path": str(audio_path),
            "gaze_path": str(gaze_path),
            "smirk_path": str(smirk_path),
            "motion_len": motion_len,
            "audio_len": audio_len,
            "gaze_len": len(gaze),
            "smirk_path": len(smirk),
        })
    except Exception as e:
        print(f"[Skip] {motion_path.name}: {e}")

with open("dataset_meta.json", "w") as f:
    json.dump(meta, f)