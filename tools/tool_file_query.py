from langchain_core.tools import tool
import os

@tool
def read_local_file(file_path:str) -> str:
    """
    读取本地磁盘上txt、md文件内容，查询文件内部信息
    Args:
        file_path: 文件的相对路径，例如 ./doc/test.txt
    """
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"
    suffix = file_path.lower().split(".")[-1]
    try:
        if suffix in ["txt","md"]:
            with open(file_path,"r",encoding="utf-8") as f:
                content = f.read()
            return f"文件内容:\n{content[:1500]}"
        else:
            return "当前工具仅支持txt、md文件读取"
    except Exception as e:
        return f"读取文件失败,错误:{str(e)}"