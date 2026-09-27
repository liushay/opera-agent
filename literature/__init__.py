# literature 文献生成模块
# 负责戏曲文献的生成与文件输出（支持txt/pdf/md三种格式）
# 文献统一存放至独立根目录，根目录内按文件类型分子文件夹

from .generator import LiteratureGenerator, literature_generator

__all__ = ["LiteratureGenerator", "literature_generator"]