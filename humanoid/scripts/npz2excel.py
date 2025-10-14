import numpy as np
import pandas as pd

# 读取npz文件
npz_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_aligned_npz/slam_new/partitioned_aligned_slam_1.npz"
data = np.load(npz_path, allow_pickle=True)
print(data.files)
# 创建Excel writer
excel_path = "/home/zl/Downloads/kuavo-rl-train-run/RL_train/output_data/partitioned_aligned_npz/slam_new/partitioned_aligned_slam_1.npz".replace(".npz", ".xlsx")
with pd.ExcelWriter(excel_path) as writer:
    for key in list(data.files)[:3]:
        arr = data[key]
        # 如果是一维，转成二维
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        df = pd.DataFrame(arr)
        df.to_excel(writer, sheet_name=key, index=False)

print(f"已保存为 {excel_path}")
