from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocProcessor:
    def __init__(self, chunk_size=120, chunk_overlap=15):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " "]
        )

    def load_file(self, file_path: str):
        # txt、md都用轻量TextLoader读取，无额外依赖
        if file_path.endswith(".txt") or file_path.endswith(".md"):
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            raise Exception("仅支持txt/md/pdf格式")
        return loader.load()

    def split_docs(self, docs):
        all_chunks = []
        for doc in docs:
            chunks = self.splitter.split_text(doc.page_content)
            for chunk in chunks:
                meta = doc.metadata or {}  # 关键修复，None替换为空字典
                all_chunks.append({
                    "text": chunk,
                    "source": meta
                })
        return all_chunks

if __name__ == "__main__":
    processor = DocProcessor(chunk_size=100, chunk_overlap=10)
    raw_docs = processor.load_file("test.pdf")
    chunk_list = processor.split_docs(raw_docs)
    print("统一处理后的文本分块：")
    for item in chunk_list:
        print(f"文本：{item['text']} 来源：{item['source']}")