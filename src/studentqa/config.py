"""配置管理"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    # OpenAI 配置
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    # 模型配置
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    LLM_MODEL: str = "gpt-4o-mini"

    # 路径配置
    PDF_PATH: str = "data/成都信息工程大学学生手册.pdf"
    VECTOR_DB_PATH: str = "./vector_db"

    # 检索配置
    TOP_K: int = 5
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # 向量数据库配置
    COLLECTION_NAME: str = "student_handbook"
    OLLAMA_HOSTS: list[str] = ["localhost:11434", "127.0.0.1:11434"]

    # API 配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # LangSmith 配置
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: bool = False

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()