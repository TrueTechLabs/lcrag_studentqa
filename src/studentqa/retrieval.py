"""向量存储与检索"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool

from .config import settings


# 全局单例
_embeddings = None
_vector_store = None


def get_embeddings():
    """获取嵌入模型单例"""
    global _embeddings
    if _embeddings is None:
        # 根据配置判断使用 Ollama 还是 OpenAI
        if any(host in settings.OPENAI_API_BASE for host in settings.OLLAMA_HOSTS):
            _embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        else:
            _embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
            )
    return _embeddings


def get_vector_store():
    """获取向量存储单例"""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=settings.COLLECTION_NAME,
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=get_embeddings(),
        )
    return _vector_store

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """检索文档上下文用于回答问题"""
    vector_store = get_vector_store()
    embedding = get_embeddings().embed_query(query)
    retrieved_docs = vector_store.similarity_search_by_vector(embedding, k=settings.TOP_K)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    print(f"[RETRIEVE] Retrieved {len(retrieved_docs)} documents for query: {query}", flush=True)

    return serialized, retrieved_docs