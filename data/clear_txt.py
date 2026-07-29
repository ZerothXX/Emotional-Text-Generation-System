import os


def process_txt_files(input_folder="corpus1", output_file="corpus.txt"):
    specific_prefix = "旁白："
    if not os.path.isdir(input_folder):
        print(f"错误：文件夹 '{input_folder}' 不存在。请创建该文件夹并放入您的txt文件。")
        return

    with open(output_file, 'a', encoding='utf-8') as outfile:
        print(f"开始处理 '{input_folder}' 文件夹中的文件...")

        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                input_filepath = os.path.join(input_folder, filename)
                # try:
                with open(input_filepath, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        stripped_line = line.rstrip()
                        processed_line = None
                        if stripped_line.startswith(specific_prefix):
                            continue
                            # processed_line = stripped_line[len(specific_prefix):].strip()
                        elif ":" in stripped_line:
                            first_colon_index = stripped_line.find(":")
                            processed_line0 = stripped_line[first_colon_index + 1:].strip()
                            if processed_line0:
                                if processed_line0[0] == "「" or processed_line0[0] == "『" or processed_line0[0] == "[" or \
                                        processed_line0[0] == "【":
                                    processed_line = processed_line0[1:-1]
                                else:
                                    processed_line = processed_line0
                        elif "：" in stripped_line:
                            first_colon_index = stripped_line.find("：")
                            processed_line0 = stripped_line[first_colon_index + 1:].strip()
                            if processed_line0:
                                if processed_line0[0] == "「" or processed_line0[0] == "『" or processed_line0[0] == "[" or \
                                        processed_line0[0] == "【":
                                    processed_line = processed_line0[1:-1]
                                else:
                                    processed_line = processed_line0
                        else:
                            processed_line = stripped_line
                            # continue  # xttxt文件夹使用
                        if processed_line is not None:
                            outfile.write(processed_line + '\n')
                        else:
                            continue
                print(f"文件 {filename} 处理完毕。")
                # except Exception as e:
                #     print(f"处理文件 {filename} 时发生错误: {e}")
    print(f"\n所有文件处理完毕。结果已写入 '{output_file}' 文件。")


# 调用函数开始执行
if __name__ == "__main__":
    process_txt_files()
