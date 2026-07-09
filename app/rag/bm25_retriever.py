from langchain_core.documents import Document
import jieba
from app.config import settings
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from langchain_community.retrievers import BM25Retriever
from typing import Any

def load_bm25_documents()-> list[Document]:
    """加载并切分文档，给 BM25 检索器使用。"""
    documents  = load_documents(settings.raw_data_path)

    if  not documents:
        raise ValueError("没有找到任何文档")
    chunks = split_documents(documents)
    return chunks

def tokenize_for_bm25(text: str)-> list[str]:
    """使用 jieba 分词对文本进行分词。"""
    text = text.strip().lower()

    if not text:
        return []

    word_tokens = [token for token in jieba.lcut(text) if token.strip()]
    #“补充保险”单字
    char_tokens = [token for token in text.replace(" ","") if token.strip()]

    return word_tokens + char_tokens

def build_bm25_retriever():
    """创建并配置 BM25 检索器。"""
    chunks = load_bm25_documents()

    if not chunks:
        raise ValueError("没有找到任何文档")

    retrievers = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=tokenize_for_bm25,

    )
    return retrievers






