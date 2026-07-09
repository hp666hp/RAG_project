import argparse

from app.config import settings
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import (
    add_documents_to_vector_store,
    reset_vector_store,
    similarity_search_with_score,
)


def ingest_documents(reset: bool = False):
    print("开始执行离线建库流程")
    print(f"原始文档目录：{settings.raw_data_path}")
    print(f"Chroma 保存目录：{settings.chroma_db_path}")

    if reset:
        reset_vector_store()

    documents = load_documents(settings.raw_data_path)
    if not documents:
        print("未在 data/raw 中读取到可处理文档，请先准备 .md 或 .txt 文件。")
        return

    print(f"已读取原始文档数量：{len(documents)}")
    chunks = split_documents(documents)
    print(f"已切分文档数量：{len(chunks)}")
    add_documents_to_vector_store(chunks)

    print("文档已成功写入 Chroma")
    results = similarity_search_with_score("商品签收后还能申请退货吗？", top_k=1)

    if results:
        print("检索测试成功，最相关 chunk：")
        top_document, top_score = results[0]
        print(top_document.page_content[:200])
        print(top_document.metadata)
        print(f"相似度分数：{top_score}")
    else:
        print("检索测试没有返回结果")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 data/raw 中的文档写入 Chroma")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="写入前先清空旧的 Chroma collection",
    )

    args = parser.parse_args()
    ingest_documents(reset=args.reset)
