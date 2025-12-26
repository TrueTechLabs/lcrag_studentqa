"""启动入口"""
import sys
from pathlib import Path

import uvicorn

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from studentqa.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "studentqa.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )