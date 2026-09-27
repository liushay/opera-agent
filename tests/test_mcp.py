# tests/test_mcp.py MCP Server 工具测试
# 运行：python -m tests.test_mcp
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.rag_tools_mcp import RagToolsMcpServer


def test_mcp_initialize():
    """验证 initialize 握手"""
    srv = RagToolsMcpServer()
    resp = srv.handle_message({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"},
    })
    assert resp["result"]["serverInfo"]["name"] == "rag-tools"
    assert resp["result"]["capabilities"]["tools"] == {}
    print("  [PASS] MCP initialize")


def test_mcp_tools_list():
    """验证 tools/list 返回4个工具"""
    srv = RagToolsMcpServer()
    resp = srv.handle_message({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list",
    })
    tools = resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert "kb_search" in names
    assert "doc_get" in names
    assert "lit_generate" in names
    assert "eval_rag" in names
    assert len(tools) == 4
    print("  [PASS] MCP tools/list（4个工具）")


def test_mcp_kb_search_args():
    """验证 kb_search 参数校验"""
    srv = RagToolsMcpServer()
    resp = srv.handle_message({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "kb_search", "arguments": {"query": ""}},
    })
    # 空查询返回错误
    text = resp["result"]["content"][0]["text"]
    assert "query 不能为空" in text
    print("  [PASS] MCP kb_search 空参数校验")


def test_mcp_doc_get_safety():
    """验证 doc_get 路径安全校验"""
    srv = RagToolsMcpServer()
    resp = srv.handle_message({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "doc_get", "arguments": {"path": "../../etc/passwd"}},
    })
    text = resp["result"]["content"][0]["text"]
    assert "非法路径" in text or "禁止越权" in text
    print("  [PASS] MCP doc_get 路径安全校验")


def test_mcp_unknown_tool():
    """验证未知工具返回错误"""
    srv = RagToolsMcpServer()
    resp = srv.handle_message({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "not_exist", "arguments": {}},
    })
    assert resp["error"]["code"] == -32601
    print("  [PASS] MCP 未知工具错误")


def test_mcp_doc_get_real():
    """验证 doc_get 读取真实知识库文件"""
    kb_file = "./knowledge_base/opera/京剧_脸谱艺术研究.txt"
    if os.path.exists(kb_file):
        srv = RagToolsMcpServer()
        resp = srv.handle_message({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "doc_get", "arguments": {"path": kb_file}},
        })
        text = resp["result"]["content"][0]["text"]
        assert "京剧脸谱" in text
        print("  [PASS] MCP doc_get 读取知识库文件")
    else:
        print("  [SKIP] 知识库文件不存在，跳过")


if __name__ == "__main__":
    print("=" * 50)
    print("测试 MCP Server")
    print("=" * 50)
    test_mcp_initialize()
    test_mcp_tools_list()
    test_mcp_kb_search_args()
    test_mcp_doc_get_safety()
    test_mcp_unknown_tool()
    test_mcp_doc_get_real()
    print("=" * 50)
    print("test_mcp.py 全部通过！")
    print("=" * 50)