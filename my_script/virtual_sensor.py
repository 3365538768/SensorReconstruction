import numpy as np

def generate_time_series(n_steps: int, values_per_step: int = 10000) -> np.ndarray:
    """
    生成形状为 (n_steps, values_per_step) 的随机浮点数数组，
    每个值在 [0,1) 范围内并保留 4 位小数。
    """
    data = np.random.rand(n_steps, values_per_step)
    data = np.round(data, 4)
    return data

def save_series_to_txt(data: np.ndarray, filename: str):
    """
    将二维数组保存为纯文本文件，保留 4 位小数，空格分隔。
    
    参数:
        data (np.ndarray): 要保存的数据，形状为 (n_steps, values_per_step)
        filename (str): 输出文件路径，例如 'series.txt'
    """
    # fmt='%.4f' 表示保留 4 位小数
    # delimiter=' ' 表示以空格分隔
    np.savetxt(filename, data, fmt='%.4f', delimiter=' ')

if __name__ == "__main__":
    n_steps = 220                # 人为设定的时间步数
    series = generate_time_series(n_steps)
    
    output_file = "/users/lshou/4DGaussians/sensor/sensor_data.txt"
    save_series_to_txt(series, output_file)
    print(f"已生成 {n_steps} 行，每行 10000 个数据，保存在 `{output_file}`。")
