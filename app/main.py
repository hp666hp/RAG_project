from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from app.rag.chain import answer_question
from app.config import settings
from app.rag.loader import SUPPORTED_EXTENSIONS, load_documents
from pathlib import Path
from app.rag.splitter import split_documents
from app.rag.vector_store import add_documents_to_vector_store, reset_vector_store

# 初始化FastAPI 应用实例
app = FastAPI(
    title="RAG 知识问答系统 API",
    description="""
    基于 FastAPI 的 RAG 知识问答服务
    
    主要功能接口：
    - 根路径: GET /
    - 健康检查: GET /health
    - RAG 问答: POST /ask
    - 文档列表: GET /documents
    - 上传文档: POST /documents/upload
    - 重建知识库: POST /knowledge/rebuild
    - 删除文档: DELETE /documents/{filename}
    """,
    version="0.1.0",
)


class AskRequest(BaseModel):
    """问答请求数据模型"""
    question: str = Field(..., min_length=1, description="用户提问内容")
    top_k: int|None  = Field(default=None, ge=1, le=20, description="检索返回的最相关文档片段数量")
    use_rerank: bool = True
    rerank_top_n: int = 3


@app.get("/")
def root():
    """根路径，返回 API 基本信息"""
    return {
        "message": "RAG 知识问答系统正在运行",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
    }


@app.get("/health")
def health():
    """健康检查接口"""
    return {
        "status": "ok",
        "message": "RAG API 服务正常",
    }

class SourceItem(BaseModel):
    """引用来源数据模型"""
    id: int
    source: str
    content: str
    metadata : dict = Field(default_factory=dict, description="文档元数据")


class AskResponse(BaseModel):
    """问答响应数据模型"""
    answer: str
    sources: list[SourceItem]

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    执行知识问答
    
    接收用户问题，通过 RAG 流程检索相关知识并生成回答，同时返回引用来源
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        result = answer_question(question, request.top_k, request.use_rerank, request.rerank_top_n)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理问题时发生错误: {exc}")


class DocumentItem(BaseModel):
    """文档信息数据模型"""
    file_name: str
    path: str
    file_type: str

class DocumentListResponse(BaseModel):
    """文档列表响应数据模型"""
    documents: list[DocumentItem]
    count: int


@app.get("/documents", response_model=DocumentListResponse)
def list_documents():
    """
    获取已上传的文档列表
    
    遍历原始数据目录，返回所有支持格式的文件信息
    """
    # 确保数据目录存在
    settings.raw_data_path.mkdir(parents=True, exist_ok=True)

    documents = []
    # 递归查找所有文件
    for path in sorted(settings.raw_data_path.rglob("*")):
        # 过滤支持的文件类型
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append({
                "file_name": path.name,
                "path": str(path),
                "file_type": path.suffix.lower(),
            })

    return {
        "documents": documents,
        "count": len(documents),
    }


class UploadResponse(BaseModel):
    """上传响应数据模型"""
    message: str
    file_name: str
    path: str
    size: int


@app.post("/documents/upload", response_model=UploadResponse)
def upload_document(uploaded_file: UploadFile = File(...)):
    """
    上传文档到服务器
    
    验证文件类型后，将文件保存到原始数据目录
    """
    # 获取文件扩展名
    suffix = Path(uploaded_file.filename).suffix.lower()

    # 验证文件类型是否支持
    if suffix not in sorted(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix}, 支持的类型: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    
    # 确保保存目录存在
    settings.raw_data_path.mkdir(parents=True, exist_ok=True)
    save_path = settings.raw_data_path / uploaded_file.filename

    # 读取并保存文件内容
    content = uploaded_file.file.read()
    save_path.write_bytes(content)

    return {
        "message": "文件上传成功",
        "file_name": uploaded_file.filename,
        "path": str(save_path),
        "size": len(content),
    }


class RebuildResponse(BaseModel):
    """重建知识库响应数据模型"""
    message: str
    document_count: int
    chunk_count: int

# 内部辅助函数
def rebuild_knowledge_base():
    """
    重建向量知识库
    
    1. 加载所有文档
    2. 清空现有向量库
    3. 分割文档
    4. 添加到向量存储
    """
    docs = load_documents(settings.raw_data_path)

    reset_vector_store()

    if not docs:
        return {
            "message": "没有找到可处理的文档，请先上传文档",
            "document_count": 0,
            "chunk_count": 0,
        }

    chunks = split_documents(docs)
    add_documents_to_vector_store(chunks)

    return {
        "message": "知识库重建成功",
        "document_count": len(docs),
        "chunk_count": len(chunks),
    }

# 重建知识库接口
@app.post("/knowledge/rebuild", response_model=RebuildResponse)
def rebuild_knowledge():
    """
    触发知识库重建
    
    重新加载所有文档并更新向量索引
    """
    try:
        return rebuild_knowledge_base()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建知识库失败: {e}")


class DeleteResponse(BaseModel):
    """删除响应数据模型"""
    message: str
    file_name: str


@app.delete("/documents/{filename}", response_model=DeleteResponse)
def delete_document(filename: str):
    """
    删除指定文档
    
    从磁盘删除文件，注意：此操作不会自动更新向量库，建议删除后重建知识库
    """
    raw_data_dir = settings.raw_data_path.resolve()
    file_path = (raw_data_dir / filename).resolve()

    # 安全检查：防止路径遍历攻击
    if raw_data_dir not in file_path.parents:
        raise HTTPException(status_code=400, detail="非法的文件路径")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="路径不是有效文件")

    file_path.unlink()

    return {
        "message": "文件删除成功，建议重建知识库以生效",
        "file_name": filename,
    }
