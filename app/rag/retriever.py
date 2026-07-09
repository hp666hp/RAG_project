from langchain_classic.retrievers import EnsembleRetriever
from app.rag.bm25_retriever import build_bm25_retriever

from app.rag.vector_store import get_vector_store

def normalize_question(question: str) -> str:
    """
    清洗并校验用户输入的问题字符串。

    该函数执行以下操作：
    1. 去除字符串首尾的空白字符（空格、换行符等）。
    2. 检查处理后的字符串是否为空。

    Args:
        question (str): 用户输入的原始问题字符串。

    Returns:
        str: 清洗后的非空问题字符串。

    Raises:
        ValueError: 如果清洗后的问题字符串为空，则抛出此异常。
    """
    # 去除首尾空白字符
    question = question.strip()
    
    # 校验问题是否为空
    if not question:
        raise ValueError("问题不能为空")

    return question


def build_vector_retriever(top_k: int = 3):
    """
    基于向量数据库构建 LangChain 标准的向量检索器。

    该函数获取已初始化的向量存储实例，并将其封装为 LangChain 的 Retriever 接口，
    以便在后续的检索链中使用。默认返回相似度最高的 top_k 个文档片段。

    Args:
        top_k (int, optional): 检索返回的最大文档数量。默认为 3。

    Returns:
        VectorStoreRetriever: 配置好搜索参数的 LangChain 向量检索器对象。
    """
    # 获取全局向量存储实例
    vectorstore = get_vector_store()
    
    # 将向量存储转换为 LangChain 检索器，并设置返回文档数量
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k}
    )
    
    return vector_retriever


def build_hybrid_retriever(top_k: int = 3) -> EnsembleRetriever:
    """
    构建基于 LangChain EnsembleRetriever 的混合检索器。

    该检索器结合了 BM25 算法（关键词匹配）和向量检索（语义匹配）的优势：
    - BM25 检索器：擅长精确匹配特定术语。
    - 向量检索器：擅长理解语义相似性。
    
    通过加权融合两者的结果，以提高检索的全面性和准确性。
    当前权重设置为：BM25 占 40%，向量检索占 60%。

    Args:
        top_k (int, optional): 每个子检索器返回的候选文档数量上限。默认为 3。

    Returns:
        EnsembleRetriever: 配置好权重和子检索器的混合检索器对象。
    """
    # 初始化 BM25 稀疏检索器
    bm25_retriever = build_bm25_retriever()
    
    # 初始化向量稠密检索器
    vector_retriever = build_vector_retriever(top_k=top_k)

    # 创建集成检索器，组合两种检索方式
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],  # BM25权重0.4，向量权重0.6
    )
    
    return hybrid_retriever



