import numpy as np
from pathlib import Path

def remove_short_npy_files(folder_path, min_length=2):
    folder = Path(folder_path)
    npy_files = list(folder.glob("*.npy"))

    print(f"[Info] Found {len(npy_files)} .npy files in {folder}")
    removed = 0

    for npy_file in npy_files:
        try:
            data = np.load(npy_file, mmap_mode='r')
            length = len(data)
            if length < min_length:
                print(f"[Remove] {npy_file} (length={length})")
                npy_file.unlink()
                removed += 1
        except Exception as e:
            print(f"[Error] Failed to read {npy_file}: {e}")

    print(f"[Done] Removed {removed} short .npy files.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python remove_short_npy.py /path/to/npy_folder")
    else:
        remove_short_npy_files(sys.argv[1])
