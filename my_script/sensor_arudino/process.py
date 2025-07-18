import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 新增：用于画热力图

def interpolate_csv(input_path: str, output_path: str, n: int):
    """
    将 input_path 中的 m 行帧数据插值扩展到 n 行，保存到 output_path，
    并输出一个“帧-传感器”热力图到同路径下的 *_heatmap.png。
    假设第一列名为 'frame'，后面是 v00, v01, … vRC 列。
    """
    # 1. 读入原始 CSV
    df = pd.read_csv(input_path)
    cols = df.columns.tolist()
    if cols[0] != 'frame':
        raise ValueError("CSV 第一列必须是 'frame'")

    m = len(df)
    if n <= m:
        raise ValueError(f"目标帧数 n={n} 必须大于原始帧数 m={m}")

    # 2. 构造原始与目标的 x 轴
    x_orig = np.arange(m)
    x_new  = np.linspace(0, m - 1, n)

    # 3. 准备输出表
    df_out = pd.DataFrame({'frame': np.arange(n)})

    # 4. 对每个传感器列做线性插值
    for col in cols[1:]:
        y_orig = df[col].astype(float).values
        y_new  = np.interp(x_new, x_orig, y_orig)
        df_out[col] = np.round(y_new, 2)

    # 5. 保存插值结果
    df_out.to_csv(output_path, index=False)
    print(f"插值完成：原始 {m} 帧 → 目标 {n} 帧，已写入 {output_path}")

    # 6. 生成并保存热力图
    #    用 df_out 除 frame 外的所有列，构建 (n × P) 的数值矩阵
    data = df_out.iloc[:, 1:].values  # shape: (n, P)
    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', interpolation='nearest')
    plt.colorbar(label='sensor voltage')
    plt.xlabel('sensor index')
    plt.ylabel('frame')
    plt.title('interpolation heatmap')
    # 构造热力图文件名：在 output_path 的基础上加后缀
    heatmap_path = output_path.rsplit('.', 1)[0] + '_heatmap.png'
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"热力图已保存：{heatmap_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="将传感器 CSV 中的 m 帧数据插值扩展到 n 帧，并输出热力图"
    )
    p.add_argument("input_csv",  help="原始 CSV 路径")
    p.add_argument("output_csv", help="插值后 CSV 保存路径")
    p.add_argument("n",          type=int, help="目标帧数 (n > m)")
    args = p.parse_args()

    interpolate_csv(args.input_csv, args.output_csv, args.n)