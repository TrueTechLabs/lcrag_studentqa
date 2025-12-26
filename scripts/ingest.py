"""数据入库脚本"""
import sys
from pathlib import Path

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from studentqa.loader import ingest_documents, test_retrieval


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_retrieval()
    else:
        ingest_documents()