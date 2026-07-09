from app.config import settings

"""负责加载本地 cross-encoder 模型"""
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
"""负责“把基础检索器 + reranker” 包成一个新的 retriever"""
from langchain_classic.retrievers import ContextualCompressionRetriever
"""负责“拿模型给文档重新打分”"""
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker)

import torch
from app.rag.retriever import build_hybrid_retriever, normalize_question
from langchain_core.documents import Document

def load_reranker_model(
    model_name_path: str = settings.reranker_model_path,
):
    """
    加载 HuggingFace CrossEncoder 重排序模型。

    Args:
        model_name_path (str): 重排序模型的名称或路径，默认为 "BAAI/bge-reranker-base"。

    Returns:
        HuggingFaceCrossEncoder: 初始化后的 CrossEncoder 模型实例。
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HuggingFaceCrossEncoder(
        model_name=model_name_path,
        model_kwargs={"device": f"{device}"},
    )
    return model


def build_rerank_retriever(
    top_k: int = 3,
    rerank_top_n: int = 3,
    model_name_path: str = settings.reranker_model_path,
):
    """
    构建带重排序能力的检索器。

    该函数首先创建一个混合检索器获取初步结果，然后使用 CrossEncoder 模型
    对这些结果进行重排序，最终返回一个经过上下文压缩的检索器。

    Args:
        top_k (int): 基础检索器初始召回的文档数量，默认为 3。
        rerank_top_n (int): 重排序后保留的最终文档数量，默认为 3。
        model_name_path (str): 重排序模型的名称或路径，默认为 "BAAI/bge-reranker-base"。

    Returns:
        ContextualCompressionRetriever: 包含重排序逻辑的检索器实例。
    """
    # 1. 构建基础混合检索器
    base_retriever = build_hybrid_retriever(top_k=top_k)

    # 2. 加载重排序模型
    reranker_model = load_reranker_model(model_name_path=model_name_path)

    # 3. 创建文档压缩器（重排序器）
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=rerank_top_n,
    )

    # 4. 将基础检索器和压缩器组合成最终的检索器
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    return retriever


def retriever_top_k(
        question: str,
        top_k: int = 3,
        use_rerank: bool = True,
        rerank_top_n: int = 3,
) -> list[Document]:
    """检索主入口：支持混合检索和重排序。"""
    question = normalize_question(question)

    if use_rerank:
        if rerank_top_n > top_k:
            raise ValueError("rerank_top_n 必须小于等于  top_k.")
        retriever = build_rerank_retriever(
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )

    else:
        retriever = build_hybrid_retriever(top_k=top_k)

    documents = retriever.invoke(question)

    return documents