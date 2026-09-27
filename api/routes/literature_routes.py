# api/routes/literature_routes.py 戏曲文献生成接口路由
# 包含：生成 / 列表 / 读取 / 下载 / 状态查询
# 改造点（问题1修复）：
#   1. 接口级超时限制：使用 asyncio.wait_for + 线程池执行同步LLM调用，超时返回明确错误，避免无限挂起
#   2. 执行状态标记：生成开始即写入状态（pending/running/success/error/timeout），前端可通过状态接口感知
#   3. 保留原有输出格式、返回字段（新增 task_id/status 附加字段，不影响旧调用方）
import os
import asyncio
import uuid
import threading
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

import config
from api.schema import CommonResponse, LiteratureGenerateRequest
from literature.generator import literature_generator
from utils.logger import log_error, log_info, log_warn
from utils.rag_exceptions import DocProcessException, LLMModelException

# 路由前缀
router = APIRouter(prefix="/api/literature", tags=["戏曲文献生成"])

# ===================== 生成状态标记 =====================
# 全局生成任务状态表：task_id -> {status, msg, files, created_at, finished_at}
# 线程安全：dict 原子操作 + 只在 GIL 内更新，简单场景足够
_generation_status: Dict[str, Dict] = {}
_status_lock = threading.Lock()


def _init_generation_status() -> Dict[str, Dict]:
    """初始化生成状态表"""
    return {
        "status": "idle",
        "msg": "暂无生成任务",
        "task_id": "",
        "files": {},
        "root_dir": config.LITERATURE_ROOT_DIR,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": "",
    }


def _update_status(task_id: str, **kwargs):
    """线程安全更新任务状态"""
    with _status_lock:
        if task_id not in _generation_status:
            _generation_status[task_id] = {
                "status": "pending",
                "msg": "任务已创建",
                "files": {},
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "finished_at": "",
            }
        _generation_status[task_id].update(kwargs)


def _finish_status(task_id: str, status: str, msg: str, files: Optional[dict] = None):
    """标记任务完成（success/error/timeout）"""
    _update_status(
        task_id,
        status=status,
        msg=msg,
        files=files or {},
        finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def _run_generate_sync(genre: str, theme: str, length: int,
                       formats: list, title: Optional[str],
                       task_id: str) -> dict:
    """在子线程中执行同步文献生成，并更新状态标记"""
    try:
        _update_status(task_id, status="running", msg="正在调用大模型生成文献内容...")
        result = literature_generator.generate_literature(
            genre=genre,
            theme=theme,
            length=length,
            formats=formats,
            title=title,
        )
        _finish_status(task_id, "success", "文献生成成功", result)
        log_info("文献状态", f"任务[{task_id}]生成完成，输出：{result}")
        return result
    except LLMModelException as e:
        _finish_status(task_id, "error", e.msg)
        log_error("文献状态", f"任务[{task_id}]LLM异常：{e.msg}", e.origin_err)
        raise e
    except DocProcessException as e:
        _finish_status(task_id, "error", e.msg)
        log_error("文献状态", f"任务[{task_id}]输出异常：{e.msg}", e.origin_err)
        raise e
    except Exception as e:
        err_msg = f"文献生成任务[{task_id}]执行异常：{str(e)}"
        _finish_status(task_id, "error", err_msg)
        log_error("文献状态", err_msg, e)
        raise e


@router.post("/generate", response_model=CommonResponse)
async def literature_generate(req: LiteratureGenerateRequest):
    """
    生成戏曲文献接口
    支持输出 txt / pdf / md 三种格式，文件保存至文献根目录按类型分子文件夹
    改造点：
      - 异步 + 线程池执行，asyncio.wait_for 设置接口级超时（LITERATURE_GENERATE_TIMEOUT）
      - 超时返回明确错误，绝不无限挂起
      - 返回中附加 task_id / status 字段（新增字段，保留原有 files / root_dir 结构）
    """
    # 生成唯一任务ID，先行标记 pending
    task_id = str(uuid.uuid4())
    _update_status(task_id, status="pending", msg="任务已创建，等待执行", files={})

    timeout = getattr(config, "LITERATURE_GENERATE_TIMEOUT", 180)
    log_info("文献生成接口", f"任务[{task_id}]开始，超时限制：{timeout}s")

    try:
        # 将同步阻塞的 generate_literature 放入线程池，避免阻塞事件循环
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _run_generate_sync,
                req.genre,
                req.theme,
                req.length,
                req.formats or ["txt", "pdf", "md"],
                req.title,
                task_id,
            ),
            timeout=timeout,
        )
        # 整理返回文件列表（与原有结构完全一致）
        file_list = []
        for fmt, path in result.items():
            file_list.append(
                {
                    "format": fmt,
                    "path": path,
                    "filename": os.path.basename(path),
                }
            )
        return CommonResponse(
            code=200,
            msg=f"戏曲文献生成成功，共输出{len(file_list)}个文件",
            data={
                "files": file_list,
                "root_dir": config.LITERATURE_ROOT_DIR,
                # 附加状态字段（不影响原有字段）
                "task_id": task_id,
                "status": "success",
            },
        )
    except asyncio.TimeoutError:
        # 超时：明确返回错误，绝不无限等待
        err_msg = f"文献生成超时（超过{timeout}秒），请减小篇幅或稍后重试"
        _finish_status(task_id, "timeout", err_msg)
        log_error("文献生成超时", err_msg, None)
        return CommonResponse(code=500, msg=err_msg, data={
            "task_id": task_id,
            "status": "timeout",
        })
    except LLMModelException as e:
        # 内部已更新状态，这里直接返回
        log_error("文献生成失败", e.msg, e.origin_err)
        return CommonResponse(code=500, msg=e.msg, data={
            "task_id": task_id,
            "status": "error",
        })
    except DocProcessException as e:
        log_error("文献输出失败", e.msg, e.origin_err)
        return CommonResponse(code=500, msg=e.msg, data={
            "task_id": task_id,
            "status": "error",
        })
    except Exception as e:
        log_error("文献生成未知异常", str(e), e)
        return CommonResponse(code=500, msg=f"文献生成失败：{str(e)}", data={
            "task_id": task_id,
            "status": "error",
        })


@router.get("/status/{task_id}", response_model=CommonResponse)
async def literature_status(task_id: str):
    """
    查询文献生成任务执行状态
    前端可通过此接口感知是否正在生成（status: pending/running/success/error/timeout）
    """
    try:
        status = _generation_status.get(task_id)
        if status is None:
            return CommonResponse(
                code=404,
                msg="任务不存在或已过期",
                data={"task_id": task_id, "status": "not_found"},
            )
        return CommonResponse(
            code=200,
            msg="状态查询成功",
            data=status,
        )
    except Exception as e:
        log_error("文献状态查询异常", str(e), e)
        return CommonResponse(code=500, msg=f"状态查询失败：{str(e)}", data={})


@router.get("/status", response_model=CommonResponse)
async def literature_status_all():
    """查询最近一次生成任务状态（兼容前端简单轮询）"""
    try:
        if not _generation_status:
            return CommonResponse(
                code=200, msg="暂无生成任务", data=_init_generation_status()
            )
        # 返回最新创建的任务状态
        latest_task_id = max(
            _generation_status.keys(),
            key=lambda tid: _generation_status[tid].get("created_at", ""),
        )
        return CommonResponse(
            code=200,
            msg="状态查询成功",
            data=_generation_status[latest_task_id],
        )
    except Exception as e:
        log_error("文献状态查询异常", str(e), e)
        return CommonResponse(code=500, msg=f"状态查询失败：{str(e)}", data={})


@router.get("/list", response_model=CommonResponse)
async def literature_list():
    """列出文献根目录下所有已生成文件，按类型分组"""
    try:
        data = literature_generator.list_literature_files()
        return CommonResponse(
            code=200,
            msg=f"共找到{data['total']}个文献文件",
            data=data,
        )
    except Exception as e:
        log_error("文献列表异常", str(e), e)
        return CommonResponse(code=500, msg=f"获取文献列表失败：{str(e)}", data={})


@router.get("/read", response_model=CommonResponse)
async def literature_read(path: str):
    """读取文献文件内容（文本预览），仅支持txt/md"""
    try:
        content = literature_generator.read_literature_content(path)
        return CommonResponse(code=200, msg="读取成功", data={"content": content})
    except DocProcessException as e:
        log_error("文献读取失败", e.msg, e.origin_err)
        return CommonResponse(code=400, msg=e.msg, data={})


@router.get("/download")
async def literature_download(path: str):
    """下载文献文件（txt/pdf/md），支持浏览器直接下载"""
    try:
        root = config.LITERATURE_ROOT_DIR
        full_path = os.path.normpath(os.path.join(root, path))
        # 安全校验：禁止越权访问
        if not full_path.startswith(os.path.normpath(root)):
            raise DocProcessException(f"非法文件路径：{path}")
        if not os.path.exists(full_path):
            raise DocProcessException(f"文件不存在：{path}")
        return FileResponse(
            path=full_path,
            filename=os.path.basename(full_path),
            media_type="application/octet-stream",
        )
    except DocProcessException as e:
        log_error("文献下载失败", e.msg, e.origin_err)
        return JSONResponse(
            status_code=400, content={"code": 400, "msg": e.msg, "data": {}}
        )
    except Exception as e:
        log_error("文献下载异常", str(e), e)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": f"下载失败:{str(e)}", "data": {}},
        )