from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """应用配置类，从 .env 文件中加载环境变量"""

    # DashScope API 配置
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    """DashScope API 密钥"""

    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL"
    )
    """DashScope API 基础 URL"""

    chat_model: str = Field(default="qwen3.7-plus", alias="CHAT_MODEL")
    """聊天模型名称"""

    embedding_model: str = Field(default="text-embedding-v4", alias="EMBEDDING_MODEL")
    """嵌入模型名称"""

    # 数据路径配置
    chroma_db_path: Path = Field(default=Path("./data/chroma_db"), alias="CHROMA_DB_PATH")
    """Chroma 向量数据库存储路径"""

    raw_data_path: Path = Field(default=Path("./data/raw"), alias="RAW_DATA_PATH")
    """原始数据文件存放路径"""

    # RAG 处理参数
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE", ge=100, le=4000)
    """文本分块大小"""

    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP", ge=0)
    """文本分块重叠大小"""

    top_k: int = Field(default=3, alias="TOP_K", ge=1, le=20)
    """检索时返回的最相似文档数量 K"""

    reranker_model_path: str= Field(
        default="C:/python-project/RAG_project/app/models/bge-reranker-v2-m3"
        , alias="RERANKER_MODEL_PATH2")

    # Pydantic Settings 配置，指定从 .env 文件读取配置，忽略额外字段
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 字段验证器
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info) -> int:
        """验证 chunk_overlap 必须小于 chunk_size"""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value > chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        return value

settings = Settings()
