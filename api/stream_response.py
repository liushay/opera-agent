from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from langchain_ollama import ChatOllama

import config


async def stream_llm_response(prompt:str) -> AsyncGenerator[str, None]:
    llm = ChatOllama(model=config.LLM_MODEL, temperature=config.LLM_TEMP, streaming=True)
    async for chunk in llm.astream(prompt):
        yield chunk.content