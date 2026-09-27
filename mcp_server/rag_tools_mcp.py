# mcp_server/rag_tools_mcp.py RAG 工具 MCP Server
# 功能：将知识库检索、文档获取、文献生成、RAG评测封装为 MCP Tools
# 协议：MCP (Model Context Protocol) over stdio —— JSON-RPC 2.0 格式
# 工具列表：
#   1. kb_search      : 混合检索知识库（BM25+向量→融合→精排）
#   2. doc_get        : 获取指定文档/文献内容
#   3. lit_generate   : 生成戏曲文献（txt/pdf/md）
#   4. eval_rag       : 执行 RAG 检索指标评测
# 运行方式：python -m mcp_server.rag_tools_mcp （通过 stdio 与 MCP Client 通信）
import json
import sys
import os
import traceback

# 确保项目根目录可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RagToolsMcpServer:
    """RAG 工具 MCP 服务器（JSON-RPC stdio 实现）"""

    SERVER_NAME = "rag-tools"
    SERVER_VERSION = "1.0.0"

    # ===================== 工具定义 =====================

    TOOLS = [
        {
            "name": "kb_search",
            "description": "混合检索知识库（BM25关键词 + Chroma向量融合 + Reranker精排），返回最相关的文档片段列表",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户检索查询文本"},
                    "top_k": {"type": "integer", "description": "返回文档条数", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
        {
            "name": "doc_get",
            "description": "获取指定文档内容（知识库中的戏曲文献 / 生成的文献文件）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件相对路径，如 knowledge_base/opera/京剧_脸谱艺术研究.txt 或 literature_output/txt/xxx.txt"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "lit_generate",
            "description": "生成戏曲学术文献，支持 txt/pdf/md 三种格式输出",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "genre": {"type": "string", "description": "戏曲种类，如京剧、豫剧、越剧"},
                    "theme": {"type": "string", "description": "文献主题"},
                    "length": {"type": "integer", "description": "目标字数"},
                    "formats": {"type": "array", "items": {"type": "string"}, "description": "输出格式列表"},
                },
                "required": ["genre"],
            },
        },
        {
            "name": "eval_rag",
            "description": "执行 RAG 检索指标评测（Recall@K / HitRate@K / MRR@K / NDCG@K），返回评测报告路径与指标",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rounds": {"type": "integer", "description": "评测轮次（多次取均值）", "minimum": 1, "maximum": 5},
                },
                "required": [],
            },
        },
    ]

    # ===================== 初始化 =====================

    def __init__(self):
        # 懒加载业务模块，避免导入失败导致 MCP 启动失败
        self._kb = None
        self._literature = None
        self._evaluator = None

    def _get_kb(self):
        if self._kb is None:
            from rag.vectorstore import hybrid_retrieve
            self._kb = hybrid_retrieve
        return self._kb

    def _get_literature(self):
        if self._literature is None:
            from literature.generator import literature_generator
            self._literature = literature_generator
        return self._literature

    def _get_evaluator(self):
        if self._evaluator is None:
            from evaluation.metrics_evaluator import metrics_evaluator
            self._evaluator = metrics_evaluator
        return self._evaluator

    # ===================== 工具实现 =====================

    def _handle_kb_search(self, args: dict) -> dict:
        query = args.get("query", "")
        top_k = int(args.get("top_k", 3))
        if not query:
            return {"error": "参数 query 不能为空"}
        hybrid_retrieve = self._get_kb()
        docs = hybrid_retrieve(query)
        results = []
        for i, doc in enumerate(docs[:top_k], 1):
            results.append({
                "rank": i,
                "content": doc.page_content,
                "metadata": doc.metadata or {},
            })
        return {
            "query": query,
            "total": len(results),
            "documents": results,
        }

    def _handle_doc_get(self, args: dict) -> dict:
        path = args.get("path", "")
        if not path:
            return {"error": "参数 path 不能为空"}

        # 安全校验：只允许访问项目内文件
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.normpath(os.path.join(root, path))
        if not full_path.startswith(root):
            return {"error": f"非法路径：{path}，禁止越权访问"}

        if not os.path.exists(full_path):
            return {"error": f"文件不存在：{path}"}

        # 仅支持读取文本类文件
        if not full_path.lower().endswith((".txt", ".md", ".json")):
            return {"error": f"不支持读取该类型文件（仅支持 txt/md/json）：{path}"}

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "path": path,
            "size_bytes": os.path.getsize(full_path),
            "content": content[:5000],  # 限制返回长度，避免超长
        }

    def _handle_lit_generate(self, args: dict) -> dict:
        literature_generator = self._get_literature()
        genre = args.get("genre", "京剧")
        theme = args.get("theme", "戏曲艺术特色与发展")
        length = int(args.get("length", 800))
        formats = args.get("formats") or ["txt", "pdf", "md"]
        result = literature_generator.generate_literature(
            genre=genre,
            theme=theme,
            length=length,
            formats=formats,
        )
        return {
            "genre": genre,
            "theme": theme,
            "output_files": result,
        }

    def _handle_eval_rag(self, args: dict) -> dict:
        metrics_evaluator = self._get_evaluator()
        rounds = int(args.get("rounds", 3))
        summary = metrics_evaluator.run_evaluation(rounds=rounds)
        return {
            "rounds": summary["rounds"],
            "question_count": summary["question_count"],
            "final_metrics": summary["final_metrics"],
            "report_path": summary["report_path"],
        }

    # ===================== MCP JSON-RPC 处理 =====================

    def handle_message(self, msg: dict) -> dict:
        """处理一条 JSON-RPC 消息"""
        method = msg.get("method", "")
        msg_id = msg.get("id", None)
        params = msg.get("params", {}) or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {
                            "name": self.SERVER_NAME,
                            "version": self.SERVER_VERSION,
                        },
                    },
                }
            elif method == "notifications/initialized":
                return None  # 通知无需响应
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"tools": self.TOOLS},
                }
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments", {}) or {}
                if tool_name == "kb_search":
                    result = self._handle_kb_search(tool_args)
                elif tool_name == "doc_get":
                    result = self._handle_doc_get(tool_args)
                elif tool_name == "lit_generate":
                    result = self._handle_lit_generate(tool_args)
                elif tool_name == "eval_rag":
                    result = self._handle_eval_rag(tool_args)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"未知工具: {tool_name}"},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
                    },
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"未知方法: {method}"},
                }
        except Exception as e:
            traceback.print_exc()
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": f"内部错误: {str(e)}"},
            }

    def run_stdio(self):
        """通过 stdin/stdout 运行 MCP Server（JSON-RPC over stdio）"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle_message(msg)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()


def main():
    server = RagToolsMcpServer()
    server.run_stdio()


if __name__ == "__main__":
    main()