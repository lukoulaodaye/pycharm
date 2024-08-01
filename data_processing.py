import warnings
import json
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import faiss

# 忽略特定的警告
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# 读取JSON文件
with open('train.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 每行一个JSON对象
data = [json.loads(line) for line in lines]

# 收集唯一案件描述
unique_cases = set()
for item in data:
    unique_cases.update([item['A'], item['B'], item['C']])

# 加载预训练的BERT模型和分词器
model_name = r'E:\eage\chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 将模型和分词器移到设备上（如有GPU可用则使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_sentence_embedding(sentence, max_length=512):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding[0]  # 返回第一个元素，因为句子的嵌入是一个长度为1的数组


# 准备存储嵌入向量的列表和FAISS索引
embeddings_list = []  # 存储所有嵌入向量
index_to_description = {}  # 用于存储索引到案情描述的映射
dimension = 768  # 假设嵌入向量的维度是768
index = faiss.IndexFlatL2(dimension)  # 创建FAISS索引

# 遍历唯一案件描述，为每个案件生成嵌入向量
for case_index, case in enumerate(unique_cases):
    embedding = get_sentence_embedding(case)
    embeddings_list.append({
        'case_index': case_index,
        'embedding': embedding
    })
    index_to_description[len(embeddings_list) - 1] = case

# 将嵌入向量列表转换为numpy数组
all_embeddings = np.array([item['embedding'] for item in embeddings_list])

# 将向量添加到FAISS索引中
index.add(all_embeddings)

# 保存FAISS索引到文件
faiss.write_index(index, "train0_sentence_embeddings.index")

print(f"所有唯一案件的向量已成功添加到FAISS索引，并保存到文件中。索引中包含的向量数量: {index.ntotal}")

# 加载FAISS索引
index = faiss.read_index('train0_sentence_embeddings.index')

# 创建查询向量（这里假设使用与索引中向量相同的维度）
query_vector = np.array([all_embeddings[0]]).astype('float32')

# 查询最相似的6个向量
k = 6
distances, indices = index.search(query_vector, k)
print(f"最近的 {k} 个向量的索引: {indices}")
print(f"最近的 {k} 个向量的距离: {distances}")

# 打印查询结果
for i in range(k):
    idx = indices[0][i]

    print(f"第 {i + 1} 个相似向量的索引: {idx}, 距离: {distances[0][i]}")
    print(f"案情描述: {index_to_description[idx]}")
