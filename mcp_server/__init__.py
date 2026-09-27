# mcp_server/__init__.py MCP Server 工具注册与启动
# 功能：将知识库检索、文档获取、文献生成、RAG评测封装为 MCP Tools
# 通过 stdio (JSON-RPC over stdin/stdout) 与 MCP Client 通信
from .rag_tools_mcp import RagToolsMcpServer, main

__all__ = ["RagToolsMcpServer", "main"]