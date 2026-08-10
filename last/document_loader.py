from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
import os

def load_txt(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return docs

def load_md(file_path: str):
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return docs

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return pages

if __name__ == "__main__":
    # 读取txt
    txt_docs = load_txt("../test.txt")
    print("==== TXT文档内容 ====")
    print(txt_docs[0].page_content)

    # 读取md
    md_docs = load_md("../test.md")
    print("\n==== Markdown文档内容 ====")
    print(md_docs[0].page_content)

    # 读取PDF（有文件再运行，无则注释此行）
    pdf_docs = load_pdf("../test.pdf")
    print("\n==== PDF第一页内容 ====")
    print(pdf_docs[0].page_content)