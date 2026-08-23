# api/routes/evaluation_routes.py 检索指标评测接口路由
# 包含：执行评测 / 读取评测报告
import os
from fastapi import APIRouter

import config
from api.schema import CommonResponse, EvaluationRequest
from evaluation.metrics_evaluator import metrics_evaluator
from utils.logger import log_error

# 路由前缀
router = APIRouter(prefix="/api/evaluation", tags=["检索指标评测"])


@router.post("/run", response_model=CommonResponse)
async def evaluation_run(req: EvaluationRequest):
    """执行检索指标量化评测：Recall@K / HitRate@K / MRR@K / NDCG@K"""
    try:
        result = metrics_evaluator.run_evaluation(rounds=req.rounds)
        return CommonResponse(
            code=200,
            msg="评测完成",
            data={
                "final_metrics": result["final_metrics"],
                "rounds": result["rounds"],
                "question_count": result["question_count"],
                "report_path": result["report_path"],
            },
        )
    except Exception as e:
        log_error("评测接口异常", str(e), e)
        return CommonResponse(code=500, msg=f"评测执行失败：{str(e)}", data={})


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