def clean_text_file(input_file, output_file):
    try:
        # 1. 读取原始文件内容
        # 使用 utf-8 编码以确保能正确识别特殊标点
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 2. 执行删除操作（将目标字符串替换为空）
        # 涵盖了六点省略号、英文六点以及单字符省略号
        cleaned_content = content.replace("……", "").replace("......", "").replace("…", "").replace("【", "").replace("】", "")

        # 3. 将清理后的内容写入新文件（或原文件）
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"处理完成！结果已保存至: {output_file}")

    except FileNotFoundError:
        print("错误：找不到指定的输入文件。")
    except Exception as e:
        print(f"发生错误: {e}")


# 使用示例
input_path = "corpus0.txt"  # 你的原始文件名
output_path = "corpus.txt"  # 处理后的文件名
clean_text_file(input_path, output_path)