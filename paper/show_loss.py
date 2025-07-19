import numpy as np
import matplotlib.pyplot as plt

# 参数设定
num_images = 500
num_epochs = 600
np.random.seed(42)

# 1. 生成一个整体衰减趋势（但更慢）
decay_curve = 0.9 ** np.linspace(0, 20, num_epochs)  # 比 exp 更慢地衰减

# 2. 对每张图加入高频/中频扰动 + 噪声
loss_matrix = []
for _ in range(num_images):
    base = decay_curve.copy()

    # 添加噪声：高频振动 + 噪声
    jitter = 0.05 * np.sin(np.linspace(0, 40, num_epochs) + np.random.rand() * 5)  # 高频震荡
    noise = np.random.normal(0, 0.02, num_epochs)  # 高斯噪声

    curve = base + jitter + noise
    curve = np.clip(curve, 0, 1)
    loss_matrix.append(curve)

loss_matrix = np.stack(loss_matrix, axis=0)  # shape (500, 600)

# 3. 可视化
plt.figure(figsize=(12, 8))
plt.imshow(loss_matrix, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Image Index')
plt.title('Simulated Loss Heatmap (Slow Decay & Noisy)')
plt.tight_layout()
plt.savefig("loss_heatmap_random_slow.png", dpi=300)
plt.show()