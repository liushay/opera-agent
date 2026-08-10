# 新建 test_add.py 测试增量入库
from kb_manager.chroma_kb import kb
# 首次加载
kb.add_file_increment("test.pdf")
