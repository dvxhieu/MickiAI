"""Main entry point for the FastAPI application."""

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.chat import router as chat_router
from app.models.user import init_db
from config.settings import settings

# Khởi tạo ứng dụng
app = FastAPI(title="Personal AI Agent Platform", version="0.1.0")

# 1. Khởi tạo Database (tạo bảng nếu chưa có)
@app.on_event("startup")
async def startup_event():
    init_db()

# 2. Tích hợp Prometheus Monitoring
Instrumentator().instrument(app).expose(app)

# 3. Gắn các router API
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
def root():
    return {"message": "Welcome to Personal AI Agent Platform", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)