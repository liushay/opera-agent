# rag/splitter/text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils.logger import log_info

class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    def split_documents(self, raw_docs):
        chunk_result = []
        for doc in raw_docs:
            chunks = self.splitter.split_text(doc.page_content)
            meta = doc.metadata or {}
            for chunk_text in chunks:
                chunk_result.append({
                    "text": chunk_text,
                    "source": meta
                })
        log_info("文本分割", f"原始文档分割完成，生成文本块总数：{len(chunk_result)}")
        return chunk_result

text_splitter = TextSplitter()