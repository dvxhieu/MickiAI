"""Pydantic schemas for API request/response payloads."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    reply: str


class HealthResponse(BaseModel):
    status: str


class StreamChunk(BaseModel):
    """A single SSE payload chunk."""
    delta: str