import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)
print("读取的文档数量：", len(docs))
print("切分后的文本块数量：", len(chunks))
print("第一个文本块的前200字：", chunks[0].page_content[:200] if chunks else "没有文本块")

# 加载中文嵌入模型
print("正在加载嵌入模型……")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("嵌入模型加载完成，正在构建向量索引……")

# 创建内存向量库
vectorstore = InMemoryVectorStore(embeddings)

# 将上一节切分好的文本块转换为向量并存入
document_ids = vectorstore.add_documents(chunks)

print(f"索引构建完成，已存入 {len(document_ids)} 个文本块")

# 提示词模板
prompt = ChatPromptTemplate.from_template(
    """你是一名文档问答助手。
请依据下面的资料回答问题。
如果资料不足以回答，请明确说明无法从资料中确定答案。

资料：
{context}

问题：
{question}

回答："""
)

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=2048,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"
print("\n当前问题：", question)

# 从向量库中检索最多3个相似文本块
retrieved_docs = vectorstore.similarity_search(question, k=3)
print("检索到的文本块数量：", len(retrieved_docs))

# 查看每个检索结果的前200个字符
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\n--- 第{i}个检索结果：前200个字符 ---")
    print(doc.page_content[:200])

# 把检索结果合并为上下文
docs_content = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)

print("\n合并后的上下文长度：", len(docs_content), "个字符")

# 把问题和检索资料填入提示词
formatted_prompt = prompt.format(
    question=question,
    context=docs_content
)

# 调用大模型生成回答
print("\n正在调用大模型生成回答……")
answer = llm.invoke(formatted_prompt)

# 输出回答正文
print("\n模型回答：")
print(answer.content)