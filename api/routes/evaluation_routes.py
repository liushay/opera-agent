# api/routes/evaluation_routes.py 检索指标评测接口路由
# 包含：执行评测 / 读取评测报告
import asyncio
import os
from fastapi import APIRouter

import config
from api.schema import CommonResponse, EvaluationRequest
from evaluation.metrics_evaluator import metrics_evaluator
from utils.logger import log_error, log_info

# 路由前缀
router = APIRouter(prefix="/api/evaluation", tags=["检索指标评测"])

# 评测接口最大超时时间（秒）：动态问题生成 + 多轮检索评测整体保护，超时返回明确错误不无限挂起
EVALUATION_TIMEOUT = getattr(config, "EVALUATION_RUN_TIMEOUT", 300)


@router.post("/run", response_model=CommonResponse)
async def evaluation_run(req: EvaluationRequest):
    """
    执行检索指标量化评测：Recall@K / HitRate@K / MRR@K / NDCG@K
    改造点（问题2）：
      - 不再读取config固定5题，支持传入 topic + reference_text 动态生成测评问题集
      - 未传 topic 时兼容旧调用（自动以默认主题动态生成）
      - 返回结构不变，新增 question_type 分布信息
    改造点（Bug3）：
      - 接口级超时保护：评测逻辑放入线程池 + asyncio.wait_for，超时返回明确错误
      - 动态问题生成 LLM 调用已缩短超时(45s、无重试)，失败立即规则降级
    """
    timeout = EVALUATION_TIMEOUT
    log_info("评测接口", f"开始评测（超时限制：{timeout}s）")
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                metrics_evaluator.run_evaluation,
                req.rounds,
                req.topic or "",
                req.reference_text or "",
                req.question_count,
            ),
            timeout=timeout,
        )
        return CommonResponse(
            code=200,
            msg="评测完成",
            data={
                "final_metrics": result["final_metrics"],
                "rounds": result["rounds"],
                "question_count": result["question_count"],
                "report_path": result["report_path"],
                # 新增：动态生成问题的类型分布（兼容旧调用方，不影响原有字段）
                "question_types": _collect_question_types(result),
            },
        )
    except asyncio.TimeoutError:
        err_msg = f"评测执行超时（超过{timeout}秒），可减少评测轮次或问题数量后重试"
        log_error("评测接口超时", err_msg, None)
        return CommonResponse(code=500, msg=err_msg, data={"status": "timeout"})
    except Exception as e:
        log_error("评测接口异常", str(e), e)
        return CommonResponse(code=500, msg=f"评测执行失败：{str(e)}", data={})


def _collect_question_types(result: dict) -> dict:
    """从评测结果收集问题类型分布（兼容动态生成的问题集）"""
    types_counter = {}
    question_meta = result.get("question_meta") or {}
    for q, detail in result.get("per_query_details", {}).items():
        # 动态生成的问题带 question_type 字段，旧固定集无此字段
        q_type = (question_meta.get(q) or {}).get("question_type", "unknown")
        types_counter[q_type] = types_counter.get(q_type, 0) + 1
    return types_counter


@router.get("/report", response_model=CommonResponse)
async def evaluation_report():
    """读取已生成的评测报告内容"""
    try:
        report_path = config.EVAL_REPORT_PATH
        if not os.path.exists(report_path):
            return CommonResponse(
                code=404, msg="评测报告不存在，请先执行评测", data={}
            )
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        return CommonResponse(
            code=200, msg="评测报告读取成功", data={"report": content}
        )
    except Exception as e:
        log_error("读取评测报告失败", str(e), e)
        return CommonResponse(code=500, msg=f"读取失败：{str(e)}", data={})