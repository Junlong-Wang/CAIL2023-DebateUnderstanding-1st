import jsonlines

# 读取训练集或测试集的JSONL文件
file_path = 'test_entry.jsonl'  # 将*替换为具体的文件名

# 输出文件名
output_file = 'test.json'

# 存储所有数据的列表
all_data = []

# 该代码使用“jsonlines”库读取 JSONL 文件（“file_path”）并将其内容存储在“all_data”列表中。
with jsonlines.open(file_path, 'r') as reader:
    for item in reader:
        all_data.extend(reader)

# 将所有数据写入输出文件
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(all_data)

# 输出文件test.json现在包含了所有数据