from pathlib import Path
from langchain_core.documents import Document

# 支持的文件扩展名常量，使用集合以提高查找效率
SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_file(file_path: str | Path) -> Document:
    """
    读取单个文件并转换为 LangChain Document 对象。

    Args:
        file_path: 文件路径。

    Returns:
        Document: 包含文件内容和元数据的 Document 对象。

    Raises:
        FileNotFoundError: 当文件不存在时抛出。
        ValueError: 当路径不是文件或文件格式不支持时抛出。
    """
    path = Path(file_path)

    # 检查文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    # 检查路径是否为文件
    if not path.is_file():
        raise ValueError(f"路径 {path} 不是一个文件")

    # 获取并检查文件后缀
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        support = " ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"不支持的文件格式：{suffix}，当前支持文件类型：{support}")

    # 读取文件内容
    content = path.read_text(encoding="utf-8")

    # 构建并返回 Document 对象
    return Document(
        page_content=content,
        metadata={
            "source": str(path),
            "file_name": path.name,
            "file_type": suffix.lstrip("."),
        }
    )


def load_documents(input_file: str | Path) -> list[Document]:
    """
    读取单个文件或目录中的所有支持的文本文件（.txt, .md）。

    Args:
        input_file: 单个文件路径或目录路径。

    Returns:
        list[Document]: 包含所有读取到的 Document 对象的列表。

    Raises:
        FileNotFoundError: 当文件或目录不存在时抛出。
    """
    path = Path(input_file)

    # 检查路径是否存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    # 如果是单个文件，直接加载
    if path.is_file():
        return [load_file(path)]

    # 如果是目录，递归加载所有支持的文件
    documents: list[Document] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(load_file(file_path))
    
    return documents
