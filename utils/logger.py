import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import config

# 创建日志目录
if not os.path.exists(config.LOG_SAVE_PATH):
    os.makedirs(config.LOG_SAVE_PATH, exist_ok=True)

# 日志格式：时间 | 等级 | 链路ID(可选) | 模块标签 | 内容 | 异常堆栈
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 初始化根日志
def init_logger(name="rag_service"):
    logger = logging.getLogger(name)
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    logger.setLevel(level_map.get(config.LOG_LEVEL, logging.INFO))
    logger.handlers.clear()

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)

    # 文件滚动输出
    if config.LOG_TO_FILE:
        log_file = os.path.join(config.LOG_SAVE_PATH, f"{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.LOG_MAX_SIZE_MB * 1024 * 1024,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(file_handler)
    return logger

# 全局单例日志对象
global_logger = init_logger("rag_main")

# 对外封装分级打印方法，替代原print_log
def log_debug(tag: str, content: str, traceback: Exception = None):
    msg = f"【{tag}】{content}"
    if traceback:
        global_logger.debug(msg, exc_info=traceback)
    else:
        global_logger.debug(msg)

def log_info(tag: str, content: str, traceback: Exception = None):
    msg = f"【{tag}】{content}"
    if traceback:
        global_logger.info(msg, exc_info=traceback)
    else:
        global_logger.info(msg)

def log_warn(tag: str, content: str, traceback: Exception = None):
    msg = f"【{tag}】{content}"
    if traceback:
        global_logger.warning(msg, exc_info=traceback)
    else:
        global_logger.warning(msg)

def log_error(tag: str, content: str, traceback: Exception = None):
    msg = f"【{tag}】{content}"
    if traceback:
        global_logger.error(msg, exc_info=traceback)
    else:
        global_logger.error(msg)

# 兼容旧代码临时替换print_log（后续全项目替换为log_info/log_error）
def print_log(tag: str, content: str):
    log_info(tag, content)