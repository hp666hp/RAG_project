"""文档切块工具模块。

该模块提供了基于 LangChain 的文本切分功能，支持自定义切块大小和重叠长度，
并针对中英文混合文本优化了分隔符策略。
"""
from app.config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def build_text_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """创建并配置递归字符文本切分器。

    Args:
        chunk_size (int | None): 每个文本块的最大字符数。如果为 None，则使用配置文件中的默认值。
        chunk_overlap (int | None): 相邻文本块之间的重叠字符数。如果为 None，则使用配置文件中的默认值。

    Returns:
        RecursiveCharacterTextSplitter: 配置好的文本切分器实例。
    """
    # 如果未提供参数，则从配置文件中加载默认值
    final_chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    final_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    return RecursiveCharacterTextSplitter(
        chunk_size=final_chunk_size,
        chunk_overlap=final_chunk_overlap,
        add_start_index=True,  # 在元数据中记录每个 chunk 在原文中的起始索引，便于溯源
        separators=[
            "\n\n",  # 段落分隔
            "\n",    # 换行符
            "。",     # 中文句号
            "！",     # 中文感叹号
            "？",     # 中文问号
            ".",      # 英文句号
            "!",      # 英文感叹号
            "?",      # 英文问号
            "；",     # 中文分号
            ";",      # 英文分号
            "，",     # 中文逗号
            ",",      # 英文逗号
            " ",      # 空格
            "",       # 空字符串（最后尝试逐个字符分割）
        ],
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档列表切分为较小的文本块。

    Args:
        documents (list[Document]): 待切分的原始文档列表。

    Returns:
        list[Document]: 切分后的文本块列表，已过滤掉内容为空的块。

    Raises:
        ValueError: 如果输入的文档列表为空。
    """
    if not documents:
        raise ValueError("文档列表不能为空")

    # 初始化文本切分器
    text_splitter = build_text_splitter()

    # 执行切分操作
    chunks = text_splitter.split_documents(documents)

    # 过滤掉页面内容为空或仅包含空白字符的无效 chunk
    return [chunk for chunk in chunks if chunk.page_content.strip()]


if __name__ == "__main__":
    from app.rag.loader import load_documents

    # 加载原始文档
    documents = load_documents(settings.raw_data_path)
    
    # 执行切分
    chunks = split_documents(documents)

    # 输出统计信息
    print(f"原始文档数量：{len(documents)}")
    print(f"切块数量：{len(chunks)}")

    if chunks:
        print("第一个 chunk 内容预览：")
        print(chunks[0].page_content[:300])
        print("第一个 chunk 元数据：")
        print(chunks[0].metadata)
    else:
        print("没有生成任何 chunk，请检查 data/raw 目录中是否有 .txt 或 .md 文件。")








