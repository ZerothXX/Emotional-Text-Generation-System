"""用于提取某目录下所有.txt.tg压缩文件，并解压至指定目录"""
import os
import gzip
import shutil
import re


def decompress_to_single_folder(source_directory, target_folder_name='corpus1'):
    """
    1. 遍历 source_directory 找到所有 .txt.gz
    2. 创建 target_folder_name (如果不存在)
    3. 将所有文件解压到 target_folder_name 中，并统一命名为 0.txt, 1.txt...
    """

    # 1. 设置输出路径 (在当前工作目录下创建 corpus1)
    # 如果你想指定绝对路径，可以直接修改 output_dir 的值
    output_dir = os.path.abspath(target_folder_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

    # 2. 定义自然排序 Key (确保文件处理顺序符合人类直觉)
    custom_sort_key_re = re.compile('([0-9]+)')

    def custom_sort_key(s):
        return [int(x) if x.isdigit() else x for x in custom_sort_key_re.split(s)]

    # 3. 扫描所有 .txt.gz 文件
    all_gz_files = []
    print(f"正在扫描 '{source_directory}' 下的文件...")

    for root, dirs, files in os.walk(source_directory):
        # 如果输出目录在源目录里面，跳过输出目录，防止死循环或干扰
        if os.path.commonpath([output_dir]) == os.path.commonpath([root, output_dir]):
            continue

        for file in files:
            if file.endswith('.txt.gz'):
                full_path = os.path.join(root, file)
                all_gz_files.append(full_path)

    # 排序
    all_gz_files.sort(key=custom_sort_key)
    print(f"共找到 {len(all_gz_files)} 个压缩文件。")

    # 4. 循环解压到 corpus1
    count = 0
    for gz_path in all_gz_files:
        try:
            # 定义输出文件名: corpus1/0.txt, corpus1/1.txt ...
            output_filename = f"{count}.txt"
            output_path = os.path.join(output_dir, output_filename)

            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 打印进度 (每100个打印一次，避免刷屏)
            if count % 100 == 0:
                print(f"正在处理第 {count} 个文件: {output_filename}")

            count += 1

        except Exception as e:
            print(f"[错误] 解压文件 {gz_path} 失败: {e}")

    print("=" * 30)
    print(f"全部完成！")
    print(f"所有文件已保存在: {output_dir}")
    print(f"总计生成文件数: {count}")


# ==========================================
# 配置区域
# ==========================================
if __name__ == '__main__':
    # 输入目录：这里填写包含许多子文件夹的原始路径
    source_dir = r'D:\Python All\Pytorch Study\natural_language\Final_assignment\data\v-corpus-zh'

    # 输出文件夹名称：默认保存在脚本运行目录下的 corpus1 文件夹
    output_folder = 'corpus1'

    if os.path.exists(source_dir):
        decompress_to_single_folder(source_dir, output_folder)
    else:
        print(f"找不到源目录: {source_dir}")