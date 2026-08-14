# rag/loader/document_loader.py
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from utils.rag_exceptions import DocProcessException
from utils.logger import log_info, log_error
import os

class DocumentLoader:
    """统一文档加载器，支持txt/md/pdf"""
    SUPPORT_SUFFIX = (".txt", ".md", ".pdf")

    def load(self, file_path: str):
        # 文件存在校验
        if not os.path.exists(file_path):
            err = DocProcessException(f"文件不存在：{file_path}")
            log_error("文档加载", err.msg)
            raise err
        # 格式校验
        if not file_path.endswith(self.SUPPORT_SUFFIX):
            err = DocProcessException(f"不支持的文件格式，仅支持{self.SUPPORT_SUFFIX}")
            log_error("文档加载", err.msg)
            raise err
        try:
            if file_path.endswith((".txt", ".md")):
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            docs = loader.load()
            log_info("文档加载", f"成功读取文件 {file_path}，原始文档数：{len(docs)}")
            return docs
        except Exception as e:
            err = DocProcessException(f"读取文件 {file_path} 失败", e)
            log_error("文档加载", err.msg, e)
            raise err

document_loader = DocumentLoader()