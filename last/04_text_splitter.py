from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import load_txt

# 加载文档
docs = load_txt("../test.txt")
full_text = docs[0].page_content

# 测试1：chunk_size=50，重叠10
splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks_50 = splitter1.split_text(full_text)
print("=== chunk_size=50 分块结果 ===")
for idx, chunk in enumerate(chunks_50):
    print(f"块{idx+1}: {chunk}")

# 测试2：chunk_size=150，重叠20
splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "，", " "]
)
chunks_150 = splitter2.split_text(full_text)
print("\n=== chunk_size=150 分块结果 ===")
for idx, chunk in enumerate(chunks_150):
    print(f"块{idx+1}: {chunk}")