"""FastAPI 接口"""
from datetime import datetime
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .agent import get_agent
from .config import settings

app = FastAPI(title="学生手册问答 API")


class ChatCompletionRequest(BaseModel):
    """OpenAI ChatCompletion 兼容请求"""
    model: str
    messages: list
    temperature: float | None = 0.0
    max_tokens: int | None = 1024
    stream: bool | None = False


@app.get("/v1/models")
async def list_models():
    """返回可用模型列表"""
    models = [
        {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
    ]
    return {"data": models, "object": "list"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI Chat Completions 兼容接口"""
    agent = get_agent()

    # 转换消息格式
    messages_for_agent = [
        {"type": "human" if m["role"] == "user" else "ai", "content": m["content"]}
        for m in req.messages
    ]

    # 非流式返回
    if not req.stream:
        final_text = ""
        for event in agent.stream(
            {"messages": messages_for_agent},
            stream_mode="values",
        ):
            # 遍历每条消息，格式化输出
            latest_msg = event["messages"][-1]
            # 格式化打印
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content_preview = latest_msg.content.replace("\n", " ")
            print(f"[{timestamp}] [{latest_msg.type.upper()}] {content_preview}", flush=True)

            msg = event["messages"][-1]
            if msg.type == "ai":
                final_text = msg.content

        return {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [
                {"message": {"role": "assistant", "content": final_text}, "index": 0, "finish_reason": "stop"}
            ],
        }

    # 流式返回
    async def event_generator():
        final_msg = None
        for event in agent.stream(
            {"messages": messages_for_agent},
            stream_mode="values",
        ):
            # 遍历每条消息，格式化输出
            latest_msg = event["messages"][-1]
            # 格式化打印
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content_preview = latest_msg.content.replace("\n", " ")
            print(f"[{timestamp}] [{latest_msg.type.upper()}] {content_preview}", flush=True)

            msg = event["messages"][-1]
            if msg.type == "ai":
                final_msg = msg

        if final_msg:
            data = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": final_msg.content},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )