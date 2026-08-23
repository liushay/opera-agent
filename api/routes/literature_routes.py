# api/routes/literature_routes.py 戏曲文献生成接口路由
# 包含：生成 / 列表 / 读取 / 下载
import os
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

import config
from api.schema import CommonResponse, LiteratureGenerateRequest
from literature.generator import literature_generator
from utils.logger import log_error
from utils.rag_exceptions import DocProcessException, LLMModelException

# 路由前缀
router = APIRouter(prefix="/api/literature", tags=["戏曲文献生成"])


@router.post("/generate", response_model=CommonResponse)
async def literature_generate(req: LiteratureGenerateRequest):
    """
    生成戏曲文献接口
    支持输出 txt / pdf / md 三种格式，文件保存至文献根目录按类型分子文件夹
    """
    try:
        result = literature_generator.generate_literature(
            genre=req.genre,
            theme=req.theme,
            length=req.length,
            formats=req.formats,
            title=req.title,
        )
        # 整理返回文件列表
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
            data={"files": file_list, "root_dir": config.LITERATURE_ROOT_DIR},
        )
    except LLMModelException as e:
        log_error("文献生成失败", e.msg, e.origin_err)
        return CommonResponse(code=500, msg=e.msg, data={})
    except DocProcessException as e:
        log_error("文献输出失败", e.msg, e.origin_err)
        return CommonResponse(code=500, msg=e.msg, data={})
    except Exception as e:
        log_error("文献生成未知异常", str(e), e)
        return CommonResponse(code=500, msg=f"文献生成失败：{str(e)}", data={})


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