
import json
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import faiss

# 忽略特定的警告


# 读取JSON文件
with open('train.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 每行一个JSON对象
data = [json.loads(line) for line in lines]

# 收集唯一案件描述
unique_cases = set()
for item in data:
    unique_cases.update([item['A'], item['B'], item['C']])

# 将集合转换为列表
unique_cases_list = list(unique_cases)
print(unique_cases_list[1])