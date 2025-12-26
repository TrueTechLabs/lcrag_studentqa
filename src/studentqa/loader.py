"""数据加载与向量化入库"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from .config import settings
from tqdm import tqdm


def _get_embeddings():
    """获取嵌入模型"""
    # 根据配置判断使用 Ollama 还是 OpenAI
    if any(host in settings.OPENAI_API_BASE for host in settings.OLLAMA_HOSTS):
        return OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
    else:
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )


def load_and_split_documents():
    """加载 PDF 文档并进行切片"""
    loader = PyPDFLoader(settings.PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    print(f"Split into {len(splits)} chunks")
    return splits


def ingest_documents():
    """将文档切片向量化并存入数据库"""
    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name=settings.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.VECTOR_DB_PATH,
    )

    splits = load_and_split_documents()

    # 分批添加并显示进度
    batch_size = 50
    for i in tqdm(range(0, len(splits), batch_size), desc="向量化入库"):
        batch = splits[i : i + batch_size]
        vector_store.add_documents(batch)

    print(f"Successfully ingested {len(splits)} documents")
    return vector_store


def test_retrieval():
    """测试检索功能"""
    embeddings = _get_embeddings()
    vector_store = Chroma(
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.VECTOR_DB_PATH,
        embedding_function=embeddings,
    )

    query = "计算机学院电话"
    print(f"Query: {query}")

    results = vector_store.similarity_search(query, k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")