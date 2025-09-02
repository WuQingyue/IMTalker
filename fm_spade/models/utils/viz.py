import torch
import numpy as np
import matplotlib.pyplot as plt
import os

class FeatureMapVisualizer:
    def __init__(self, save_dir='viz.jpg', mode='mean', max_channels=4):
        """
        :param save_dir: 可选路径，将可视化结果保存为图像
        :param mode: 可视化模式 ['mean', 'max', 'channel']
        :param max_channels: 如果 mode='channel'，最多可视化多少通道
        """
        self.save_dir = save_dir
        self.mode = mode
        self.max_channels = max_channels
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def __call__(self, feature_map: torch.Tensor, name="feature"):
        if not isinstance(feature_map, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor")
        if feature_map.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {feature_map.shape}")

        B, C, H, W = feature_map.shape
        for b in range(B):
            fmap = feature_map[b].detach().cpu()

            if torch.isnan(fmap).any():
                print(f"[Warning] NaN detected in sample {b}, {name}")
            if torch.isinf(fmap).any():
                print(f"[Warning] Inf detected in sample {b}, {name}")

            if self.mode == 'mean':
                img = fmap.mean(dim=0).numpy()
                self._visualize(img, f"{name}_b{b}_mean")

            elif self.mode == 'max':
                img = fmap.max(dim=0)[0].numpy()
                self._visualize(img, f"{name}_b{b}_max")

            elif self.mode == 'channel':
                for i in range(min(self.max_channels, C)):
                    img = fmap[i].numpy()
                    self._visualize(img, f"{name}_b{b}_c{i}")
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

    def _visualize(self, img: np.ndarray, title: str):
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.axis("off")
        if self.save_dir:
            save_path = os.path.join(self.save_dir, f"{title}.png")
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()