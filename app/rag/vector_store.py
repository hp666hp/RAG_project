from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from app.config import settings


def create_embedding_model():
    """
    创建基于 DashScope 的嵌入模型实例。
    
    Returns:
        DashScopeEmbeddings: 配置好的 DashScope 嵌入模型对象。
    """
    return DashScopeEmbeddings(
        model=settings.embedding_model,
        dashscope_api_key=settings.dashscope_api_key,
    )


def get_vector_store():
    """
    获取或初始化 Chroma 向量存储实例。
    
    如果向量存储已存在，则加载它；否则，使用指定的嵌入函数创建一个新的持久化存储。
    
    Returns:
        Chroma: 配置好的 Chroma 向量存储对象。
    """
    embedding_model = create_embedding_model()
    return Chroma(
        collection_name="rag_knowledge_base",
        embedding_function=embedding_model,
        persist_directory=str(settings.chroma_db_path)
    )


def add_documents_to_vector_store(chunks):
    """
    将文档块添加到向量存储中。
    
    Args:
        chunks (list): 待存入的文档块列表。如果列表为空，将抛出 ValueError。
        
    Returns:
        Chroma: 更新后的向量存储对象。
        
    Raises:
        ValueError: 当传入的 chunks 列表为空时抛出异常。
    """
    if not chunks:
        raise ValueError("chunks 列表不能为空，请检查 splitter.py 中的文档分割逻辑")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return vector_store

def reset_vector_store():
    """
    重置向量存储，删除现有的 Chroma 集合并重新初始化。

    Returns:
        Chroma: 重新初始化后的向量存储对象。
    """
    vector_store = get_vector_store()
    try:
        vector_store.delete_collection()
        print("成功删除现有的 Chroma collection")
    except ValueError:
        print("当前不存在 Chroma collection，无需删除")

def similarity_search_with_score(query: str, top_k: int = None):
    """
    执行相似度搜索并返回带分数的结果。

    Args:
        query (str): 查询字符串。
        top_k (int, optional): 返回的最相似结果数量。如果未指定，则使用配置文件中的默认值 (settings.top_k)。
        
    Returns:
        list: 包含 (Document, score) 元组的列表，按相似度排序。
    """
    vector_store = get_vector_store()
    final_top_k = top_k if top_k is not None else settings.top_k
    return vector_store.similarity_search_with_score(query, k=final_top_k)

# 测试入口
if __name__ == "__main__":
    from app.rag.loader import load_documents
    from app.rag.splitter import split_documents

    # 加载原始文档
    documents = load_documents(settings.raw_data_path)
    # 分割文档为块
    chunks = split_documents(documents)

    # 将文档块添加到向量存储
    vector_store = add_documents_to_vector_store(chunks)

    print(f"加载文档数量: {len(documents)}")
    print(f"分割后的块数量: {len(chunks)}")
    print("Chroma 向量库构建完成")
