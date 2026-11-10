import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# 准备三段待检索的文本
texts = [
    "猫通常喜欢吃鱼。",
    "小猫是一种常见的宠物。",
    "今天北京的天气晴朗。"
]

# 准备一个问题
query = "小猫喜欢吃什么？"

# 加载中文嵌入模型
print("正在加载嵌入模型……")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 将三段文本转换成向量
document_vectors = np.array(
    embeddings.embed_documents(texts)
)

# 使用同一个模型，将问题转换成向量
query_vector = np.array(
    embeddings.embed_query(query)
)

print("\n文档向量的形状", document_vectors.shape)
print("问题向量的维度", len(query_vector))
print("第一段文本向量的前8个数", document_vectors[0, :8])

# 向量已经归一化，点积等于余弦相似度
similarities = document_vectors @ query_vector

# 按相似度从高到低排序
ranking = np.argsort(similarities)[::-1]

print("\n当前问题", query)

for rank, index in enumerate(ranking, start=1):
    print(f"\n第{rank}名，相似度为 {similarities[index]:.4f}")
    print(texts[index])