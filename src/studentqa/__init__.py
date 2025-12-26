"""成都信息工程大学学生手册问答系统"""
from .api import app
from .config import settings
from .agent import get_agent

__all__ = ["app", "settings", "get_agent"]