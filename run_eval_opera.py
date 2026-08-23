# 戏曲知识库专项评测脚本
# 功能：使用戏曲主题测试问题集执行完整的Recall/HitRate/MRR/NDCG评测
# 对比改造前后（知识库扩充前后）的检索指标，验证分层记忆+知识库检索功能
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import config
from evaluation.metrics_evaluator import metrics_evaluator
from rag.vectorstore import kb
from rag.vectorstore.bm25_retriever import bm25_kb

# 戏曲知识库专项评测问题集
OPERA_EVAL_QUESTION_SET = [
    {
        "query": "京剧脸谱的色彩象征意义是什么",
        "expected_keywords": ["脸谱", "红色", "黑色", "白色", "象征"]
    },
    {
        "query": "京剧四大名旦分别是谁",
        "expected_keywords": ["梅兰芳", "程砚秋", "尚小云", "荀慧生", "四大名旦"]
    },
    {
        "query": "豫剧花木兰的经典唱段是什么",
        "expected_keywords": ["花木兰", "谁说女子不如男", "常香玉", "唱段", "豫剧"]
    },
    {
        "query": "黄梅戏天仙配讲述了什么故事",
        "expected_keywords": ["天仙配", "七仙女", "董永", "黄梅戏", "夫妻双双把家还"]
    },
    {
        "query": "昆曲牡丹亭的艺术价值",
        "expected_keywords": ["牡丹亭", "汤显祖", "杜丽娘", "昆曲", "游园惊梦"]
    },
    {
        "query": "川剧变脸的表演原理",
        "expected_keywords": ["变脸", "川剧", "脸谱", "表演", "绝技"]
    },
    {
        "query": "越剧梁山伯与祝英台的化蝶故事",
        "expected_keywords": ["梁山伯", "祝英台", "化蝶", "越剧", "十八相送"]
    },
    {
        "query": "评剧刘巧儿的艺术特点",
        "expected_keywords": ["评剧", "刘巧儿", "新凤霞", "唱腔", "新派"]
    },
    {
        "query": "秦腔的声腔特色与文化地位",
        "expected_keywords": ["秦腔", "声腔", "慷慨激昂", "梆子", "陕西"]
    },
    {
        "query": "粤剧帝女花的经典唱段",
        "expected_keywords": ["粤剧", "帝女花", "落花满天蔽月光", "南国红豆", "唱段"]
    },
]

def run_opera_eval():
    """执行戏曲专项评测，生成报告并保存"""
    print("=" * 60)
    print("戏曲知识库专项量化评测启动")
    print("=" * 60)

    # 保存原有评测配置
    original_question_set = config.EVAL_QUESTION_SET
    original_report_path = config.EVAL_REPORT_PATH
    original_rounds = config.EVAL_ROUNDS

    try:
        # 临时切换为戏曲评测问题集
        config.EVAL_QUESTION_SET = OPERA_EVAL_QUESTION_SET
        config.EVAL_REPORT_PATH = "./evaluation_report/report_opera_eval.md"
        config.EVAL_ROUNDS = 3

        # 重建BM25索引，确保混合检索完整生效
        print("重建 BM25 索引...")
        kb.rebuild_full_bm25()
        print("BM25 索引重建完成")

        # 执行评测
        summary = metrics_evaluator.run_evaluation(rounds=3)

        # 保存JSON数据
        json_path = "./evaluation_report/report_opera_eval.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"戏曲专项评测JSON已保存：{json_path}")

        print("=" * 60)
        print("戏曲专项评测完成！最终指标：")
        for k, v in summary["final_metrics"].items():
            print(f"  {k}: {v}")
        print("=" * 60)
        return summary

    finally:
        # 恢复原始配置
        config.EVAL_QUESTION_SET = original_question_set
        config.EVAL_REPORT_PATH = original_report_path
        config.EVAL_ROUNDS = original_rounds

if __name__ == "__main__":
    run_opera_eval()