# literature/generator.py 戏曲文献生成器
# 功能：
#   1. 调用LLM生成戏曲文献内容
#   2. 输出txt / pdf / md三种格式文件
#   3. 文献统一存放至独立根目录，按文件类型分子文件夹
import os
import re
from datetime import datetime
from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

import config
from utils.logger import log_info, log_warn, log_error
from utils.rag_exceptions import LLMModelException, DocProcessException
from agent.llm_utils import invoke_with_retry


class LiteratureGenerator:
    """戏曲文献生成器"""

    # 支持的输出格式
    SUPPORT_FORMATS = ("txt", "pdf", "md")

    def __init__(self):
        # 懒加载LLM，避免模块导入时连接Ollama失败导致整个服务不可启动
        self._llm = None
        # 确保文献输出根目录存在
        self._ensure_root_dirs()

    def _ensure_root_dirs(self):
        """确保文献输出根目录及按类型划分的子目录存在"""
        root = config.LITERATURE_ROOT_DIR
        os.makedirs(root, exist_ok=True)
        # 按文件类型创建子文件夹：txt/ pdf/ md/
        for sub in config.LITERATURE_SUB_DIRS.values():
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        log_info("文献目录初始化", f"文献根目录 {root} 及其子目录已就绪")

    def _get_llm(self):
        """懒加载LLM实例"""
        if self._llm is None:
            self._llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP)
        return self._llm

    def _build_prompt(self, genre: str, theme: str, length: int) -> str:
        """构建戏曲文献生成提示词"""
        return f"""
你是资深戏曲文献研究专家，请创作一篇关于【{genre}】的学术文献。

主题方向：{theme}
目标字数：约{length}字

写作要求：
1. 包含引言（戏曲背景）、主体论述（艺术特点/历史沿革/代表剧目）、结语
2. 参考真实戏曲历史与艺术形式，语言严谨学术化
3. 严格输出纯文本正文，不要输出Markdown标记，不要输出标题序号以外的额外修饰
4. 适当分段，段落清晰

请直接输出文献正文：
"""

    def _sanitize_filename(self, name: str) -> str:
        """清洗文件名为安全文件名，去除非法字符"""
        name = re.sub(r'[\\/:*?"<>|]', "_", name)
        return name.strip() or "literature"

    def _build_timestamp(self) -> str:
        """生成时间戳，用于文件命名避免冲突"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_content(self, genre: str, theme: str, length: int) -> str:
        """
        调用LLM生成戏曲文献文本内容。
        第六部分改造：使用统一 LLM 调用层（timeout=120s / 最大重试2次 / 全链路日志），
        根治"无限转圈/卡死/无报错阻塞"问题。超时或重试耗尽抛出 LLMModelException 供前端捕获。
        """
        log_info("文献生成", f"开始生成{genre}戏曲文献，主题：{theme}，目标字数：{length}")
        prompt = self._build_prompt(genre, theme, length)
        try:
            content = invoke_with_retry(
                [prompt],
                task_name="文献生成",
            )
            # 清理可能的Markdown残留（如#、*等），保留纯文本
            content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)
            content = content.replace("**", "").replace("__", "")
            log_info("文献生成", f"文献生成完成，内容长度：{len(content)}字")
            return content
        except LLMModelException:
            raise
        except Exception as e:
            err_msg = f"戏曲文献生成失败（模型调用异常），模型：{config.LLM_MODEL}，原因：{str(e)}"
            log_error("文献生成失败", err_msg, e)
            raise LLMModelException(err_msg, e)

    # ---------- 三种格式输出方法 ----------

    def save_as_txt(self, content: str, file_path: str) -> str:
        """输出为txt文本文件"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            log_info("文献输出", f"TXT文献已保存：{file_path}")
            return file_path
        except Exception as e:
            err_msg = f"TXT文献写入失败：{file_path}"
            log_error("文献输出异常", err_msg, e)
            raise DocProcessException(err_msg, e)

    def save_as_md(self, content: str, file_path: str, genre: str, theme: str) -> str:
        """输出为Markdown文件（在txt基础上增加标题结构）"""
        try:
            md_content = (
                f"# {genre}戏曲文献\n\n"
                f"> 主题：{theme}\n"
                f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"---\n\n{content}\n"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            log_info("文献输出", f"MD文献已保存：{file_path}")
            return file_path
        except Exception as e:
            err_msg = f"MD文献写入失败：{file_path}"
            log_error("文献输出异常", err_msg, e)
            raise DocProcessException(err_msg, e)

    def save_as_pdf(self, content: str, file_path: str, genre: str, theme: str) -> str:
        """输出为PDF文件（使用reportlab生成中文PDF）"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.enums import TA_CENTER

            # 注册中文字体（优先使用常见Windows/系统字体）
            font_name = self._register_chinese_font()

            doc = SimpleDocTemplate(
                file_path,
                pagesize=A4,
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2 * cm,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CNStyleTitle",
                parent=styles["Title"],
                fontName=font_name,
                fontSize=18,
                leading=24,
                alignment=TA_CENTER,
            )
            body_style = ParagraphStyle(
                "CNStyleBody",
                parent=styles["BodyText"],
                fontName=font_name,
                fontSize=12,
                leading=20,
                spaceAfter=6,
            )

            # 构建PDF内容流
            story = []
            story.append(Paragraph(f"{genre}戏曲文献", title_style))
            story.append(Spacer(1, 12))
            story.append(
                Paragraph(
                    f"主题：{theme}　|　生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    body_style,
                )
            )
            story.append(Spacer(1, 12))

            # 正文按段落分隔
            from xml.sax.saxutils import escape

            for para in content.split("\n"):
                para = para.strip()
                if para:
                    # 转义XML特殊字符，避免ReportLab解析报错
                    safe_para = escape(para)
                    story.append(Paragraph(safe_para, body_style))

            doc.build(story)
            log_info("文献输出", f"PDF文献已保存：{file_path}")
            return file_path
        except ImportError as e:
            err_msg = "PDF生成依赖reportlab未安装，请执行：pip install reportlab"
            log_error("文献输出依赖缺失", err_msg, e)
            raise DocProcessException(err_msg, e)
        except Exception as e:
            err_msg = f"PDF文献生成失败：{file_path}"
            log_error("文献输出异常", err_msg, e)
            raise DocProcessException(err_msg, e)

    def _register_chinese_font(self):
        """注册中文字体：优先检查系统常见中文字体，避免中文乱码"""
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os as _os

        font_candidates = [
            # Windows
            r"C:\Windows\Fonts\msyh.ttc",  # 微软雅黑
            r"C:\Windows\Fonts\simsun.ttc",  # 宋体
            r"C:\Windows\Fonts\simhei.ttf",  # 黑体
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            # Linux
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]

        # 检查是否已注册过
        registered_fonts = pdfmetrics.getRegisteredFontNames()

        for path in font_candidates:
            if _os.path.exists(path):
                font_name = "ChineseFont"
                if font_name not in registered_fonts:
                    pdfmetrics.registerFont(TTFont(font_name, path))
                log_info("PDF字体", f"使用中文字体：{path}")
                return font_name

        # 无中文字体时回退Helvetica（可能导致中文乱码，但保证不崩溃）
        log_warn("PDF字体", "未找到中文字体文件，PDF中文可能显示异常，建议安装中文字体")
        return "Helvetica"

    # ---------- 对外统一入口 ----------

    def generate_literature(
        self,
        genre: str = config.LITERATURE_DEFAULT_GENRE,
        theme: str = "戏曲艺术特色与发展",
        length: int = config.LITERATURE_DEFAULT_LENGTH,
        formats: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> dict:
        """
        生成戏曲文献并输出指定格式文件

        Args:
            genre: 戏曲种类（如京剧、豫剧、越剧）
            theme: 文献主题
            length: 目标字数
            formats: 输出格式列表，支持 txt/pdf/md，默认为全部三种
            title: 自定义文件名（不含扩展名），默认自动生成

        Returns:
            dict: {"txt": "路径", "pdf": "路径", "md": "路径"}
        """
        # 参数校验
        valid_formats = formats or ["txt", "pdf", "md"]
        for fmt in valid_formats:
            if fmt not in self.SUPPORT_FORMATS:
                err_msg = f"不支持的输出格式：{fmt}，仅支持 {self.SUPPORT_FORMATS}"
                log_error("文献生成参数错误", err_msg)
                raise DocProcessException(err_msg)

        if length <= 0:
            err_msg = "文献字数必须为正整数"
            log_error("文献生成参数错误", err_msg)
            raise DocProcessException(err_msg)

        # 生成正文内容（三种格式共用同一正文）
        content = self.generate_content(genre=genre, theme=theme, length=length)

        # 构建文件名（去重时间戳）
        ts = self._build_timestamp()
        safe_title = self._sanitize_filename(title) if title else f"{genre}_{theme}_{ts}"
        safe_title = f"{safe_title}_{ts}"
        base_name = safe_title.replace(" ", "_")

        # 输出根目录
        root = config.LITERATURE_ROOT_DIR
        result = {}

        # 按格式分别输出
        for fmt in valid_formats:
            sub_dir = config.LITERATURE_SUB_DIRS.get(fmt, fmt)
            target_dir = os.path.join(root, sub_dir)
            file_path = os.path.join(target_dir, f"{base_name}.{fmt}")
            if fmt == "txt":
                self.save_as_txt(content, file_path)
            elif fmt == "md":
                self.save_as_md(content, file_path, genre, theme)
            elif fmt == "pdf":
                self.save_as_pdf(content, file_path, genre, theme)
            result[fmt] = file_path

        log_info("文献生成完成", f"文献已输出至：{result}")
        return result

    # ---------- 文献管理辅助 ----------

    def list_literature_files(self) -> dict:
        """列出文献目录下所有已生成文件，按类型分组"""
        root = config.LITERATURE_ROOT_DIR
        result = {
            "root_dir": root,
            "files": {
                "txt": [],
                "pdf": [],
                "md": [],
            },
            "total": 0,
        }
        for fmt in self.SUPPORT_FORMATS:
            sub_dir = os.path.join(root, config.LITERATURE_SUB_DIRS.get(fmt, fmt))
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir), reverse=True):
                # 只统计对应扩展名文件
                if fname.endswith(f".{fmt}"):
                    fpath = os.path.join(sub_dir, fname)
                    stat = os.stat(fpath)
                    result["files"][fmt].append(
                        {
                            "filename": fname,
                            "path": fpath,
                            "size_bytes": stat.st_size,
                            "size_kb": round(stat.st_size / 1024, 2),
                            "modified_time": datetime.fromtimestamp(
                                stat.st_mtime
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
        result["total"] = sum(len(v) for v in result["files"].values())
        return result

    def read_literature_content(self, relative_path: str) -> str:
        """读取文献文本内容，用于前端预览"""
        root = config.LITERATURE_ROOT_DIR
        # 安全校验：只允许访问文献根目录内的文件
        full_path = os.path.normpath(os.path.join(root, relative_path))
        if not full_path.startswith(os.path.normpath(root)):
            err_msg = f"非法文件路径：{relative_path}，禁止越权访问"
            log_warn("文献读取", err_msg)
            raise DocProcessException(err_msg)
        if not os.path.exists(full_path):
            err_msg = f"文献文件不存在：{relative_path}"
            log_warn("文献读取", err_msg)
            raise DocProcessException(err_msg)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            err_msg = f"读取文献文件失败：{full_path}"
            log_error("文献读取异常", err_msg, e)
            raise DocProcessException(err_msg, e)


# 全局单例
literature_generator = LiteratureGenerator()