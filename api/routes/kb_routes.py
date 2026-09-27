# api/routes/kb_routes.py 知识库管理接口路由
# 包含：文档入库 / 检索测试 / 清空 / 统计 / 文件上传
import os
from fastapi import APIRouter, UploadFile, File, Request

import config
from api.schema import CommonResponse, DocIngestRequest, RetrieveRequest
from rag.vectorstore import kb, bm25_kb, hybrid_retrieve
from utils.logger import log_error, log_info
from utils.rag_exceptions import DocProcessException

# 路由前缀
router = APIRouter(prefix="/api/kb", tags=["知识库管理"])


@router.post("/ingest", response_model=CommonResponse)
async def kb_ingest(request: Request, ingest_req: DocIngestRequest):
    """文档增量入库：文件路径加载 -> 分块 -> 写入向量库+BM25索引"""
    try:
        file_path = ingest_req.file_path
        if not os.path.exists(file_path):
            return CommonResponse(code=400, msg=f"文件不存在：{file_path}", data={})

        kb.add_file_increment(file_path)
        return CommonResponse(
            code=200,
            msg=f"文档入库成功：{os.path.basename(file_path)}",
            data={"file_path": file_path},
        )
    except DocProcessException as e:
        log_error("文档入库", e.msg, e.origin_err)
        return CommonResponse(code=500, msg=e.msg, data={})
    except Exception as e:
        log_error("文档入库异常", f"未知错误：{str(e)}", e)
        return CommonResponse(code=500, msg=f"文档入库失败：{str(e)}", data={})


@router.post("/retrieve", response_model=CommonResponse)
async def kb_retrieve(request: Request, retr_req: RetrieveRequest):
    """知识库混合检索测试接口，返回检索到的文档列表"""
    try:
        docs = hybrid_retrieve(retr_req.query)
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]
        return CommonResponse(
            code=200,
            msg=f"检索成功，返回{len(results)}条文档",
            data={"query": retr_req.query, "documents": results},
        )
    except Exception as e:
        log_error("检索接口异常", str(e), e)
        return CommonResponse(code=500, msg=f"检索失败：{str(e)}", data={})


@router.post("/clear", response_model=CommonResponse)
async def kb_clear():
    """清空向量库全部数据（同时重建空BM25索引）"""
    try:
        kb.clear_kb()
        log_info("知识库管理", "向量库已清空")
        return CommonResponse(code=200, msg="向量库已清空", data={})
    except Exception as e:
        log_error("清空向量库", str(e), e)
        return CommonResponse(code=500, msg=f"清空失败：{str(e)}", data={})


@router.get("/stats", response_model=CommonResponse)
async def kb_stats():
    """知识库统计信息：向量库文档数、BM25索引大小"""
    try:
        all_data = kb.vector_store.get()
        doc_count = len(all_data["ids"])
        bm25_count = len(bm25_kb.corpus_texts) if bm25_kb.corpus_texts else 0
        return CommonResponse(
            code=200,
            msg="知识库统计",
            data={
                "vector_doc_count": doc_count,
                "bm25_doc_count": bm25_count,
                "hybrid_enabled": config.ENABLE_HYBRID_SEARCH,
                "embed_model": config.EMBED_MODEL,
                "llm_model": config.LLM_MODEL,
            },
        )
    except Exception as e:
        log_error("知识库统计异常", str(e), e)
        return CommonResponse(code=500, msg=f"统计失败：{str(e)}", data={})


@router.post("/upload", response_model=CommonResponse)
async def kb_upload(file: UploadFile = File(...)):
    """上传文件并自动入库（txt/md/pdf）"""
    try:
        # 校验格式
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in (".txt", ".md", ".pdf"):
            return CommonResponse(
                code=400,
                msg=f"不支持的文件格式：{suffix}，仅支持txt/md/pdf",
                data={},
            )

        # 保存到临时目录
        os.makedirs("./uploads", exist_ok=True)
        save_path = os.path.join("./uploads", file.filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # 入库
        kb.add_file_increment(save_path)
        return CommonResponse(
            code=200,
            msg=f"文件{file.filename}上传并入库成功",
            data={"saved_path": save_path},
        )
    except Exception as e:
        log_error("文件上传入库异常", str(e), e)
        return CommonResponse(code=500, msg=f"上传失败：{str(e)}", data={})