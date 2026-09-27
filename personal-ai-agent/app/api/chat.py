"""Chat endpoint (REST + SSE streaming)."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.agent import Agent
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, StreamChunk
from app.models.user import get_db
from app.utils.auth import verify_token

router = APIRouter()
agent = Agent()  # singleton agent instance (history lives in-memory)
logger = logging.getLogger(__name__)


def get_current_user(token: str = Depends(oauth2_scheme)):  # will define oauth2_scheme below
    """Dependency that verifies JWT and returns user dict."""
    try:
        payload = verify_token(token)
        return payload  # contains sub (user_id) etc.
    except jwt.PyJWTError as exc:
        logger.warning("Invalid token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# Simple OAuth2 password bearer (for demo; in production use proper OAuth2)
from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


# ----------------------------------------------------------------------
# REST endpoint (non-streaming)
# ----------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Receive a user message and return the agent reply.
    This endpoint does NOT stream; use /chat/stream for SSE.
    """
    logger.info("Chat request from user %s: %s", current_user.get("sub"), req.message[:50])
    reply = agent.run(req.message)
    return ChatResponse(reply=reply)


# ----------------------------------------------------------------------
# Streaming endpoint (Server-Sent Events)
# ----------------------------------------------------------------------
@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    Stream the agent reply token-by-token using Server-Sent Events.
    The client should listen to `text/event-stream`.
    """
    async def event_generator() -> AsyncIterator[bytes]:
        try:
            logger.info("Stream chat request from user %s: %s", current_user.get("sub"), req.message[:50])

            # 1. Append user message to short-term history
            agent.history.append({"role": "user", "content": req.message})
            agent.vector_memory.add(
                text=req.message,
                metadata={"role": "user", "turn": len(agent.history)},
                doc_id=f"user_{len(agent.history)}",
            )

            # 2. Tool-use loop with streaming LLM
            for _ in range(settings.max_iterations):
                # First, do a non-streaming call to detect whether tools are needed
                response = agent.llm.generate(agent.history, tools=agent.tools_schemas())
                tool_calls = agent.llm.extract_tool_calls(response)

                if tool_calls:
                    # Append assistant message with tool_use blocks
                    agent.history.append(agent.llm.build_assistant_message(response))
                    # Execute tools and append results
                    for tc in tool_calls:
                        try:
                            result_str = str(agent.tools_execute(tc["name"], tc["input"]))
                        except Exception as exc:  # pragma: no cover
                            result_str = f"Error executing tool {tc['name']}: {exc}"
                        agent.history.append(
                            agent.llm.build_tool_result_message(tc["id"], result_str)
                        )
                        # Store tool result in long-term memory (optional)
                        agent.vector_memory.add(
                            text=f"Tool {tc['name']} result: {result_str}",
                            metadata={"role": "tool_result", "tool": tc["name"]},
                            doc_id=f"tool_{len(agent.history)}",
                        )
                    # Continue loop to let LLM ingest tool results
                    continue

                # No tool calls — stream the final answer
                agent.history.append(agent.llm.build_assistant_message(response))
                final_text = ""
                async for delta in agent.llm.generate_stream(agent.history, tools=None):
                    final_text += delta
                    # Yield SSE-formatted delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n".encode("utf-8")

                # Store final answer in long-term memory
                agent.vector_memory.add(
                    text=final_text,
                    metadata={"role": "assistant", "turn": len(agent.history)},
                    doc_id=f"assistant_{len(agent.history)}",
                )
                # Send a final [DONE] event as per SSE spec
                yield f"data: {json.dumps({'delta': '', 'done': True})}\n\n".encode("utf-8")
                break
            else:
                # Max iterations exceeded
                yield f"data: {json.dumps({'delta': '\\n[Agent stopped due to max iterations.]', 'done': True})}\n\n".encode("utf-8")
        except Exception as exc:  # pragma: no cover
            logger.exception("Error in chat stream")
            yield f"data: {json.dumps({'delta': f'\\n[Error: {exc}]', 'done': True})}\n\n".encode("utf-8")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ----------------------------------------------------------------------
# Health endpoint
# ----------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")