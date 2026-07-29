"""从huggingface中下载数据集"""

from huggingface_hub import snapshot_download
import os

# 配置下载参数

DATASETS = [
    "Limour/b-corpus"
]
# 要下载的数据集列表
DOWNLOAD_DIR = "datasets"  # 数据集下载的根目录
TOKEN = ""  # 访问令牌 (如果需要)

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

for dataset_name in DATASETS:
    # 拼接完整的本地存储路径
    local_dir = os.path.join(DOWNLOAD_DIR, dataset_name)

    # 显示下载开始信息
    print(f"正在下载数据集 {dataset_name} 到 {local_dir}")

    try:
        # 使用 snapshot_download 下载数据集
        snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",  # 关键参数: 指定这是数据集
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=TOKEN
        )

        print(f"已完成 {dataset_name} 的下载\n")

    except Exception as e:
        print(f"下载 {dataset_name} 时出错: {e}\n")

print("所有数据集下载任务完成!")