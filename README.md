# 电商售后 RAG 问答系统

基于 FastAPI、LangChain 和 Chroma 构建的检索增强问答（RAG）项目，用于电商售后知识库场景下的问答、检索与评测。

项目提供从原始文档导入、文本切分、向量建库、混合检索、重排到问答接口与离线评测的一整套流程，适合作为本地可运行的 RAG 工程示例。

## 项目特性

- 基于 FastAPI 提供问答与知识库管理接口
- 使用 Chroma 构建本地向量数据库
- 支持 BM25 + 向量检索的混合召回
- 支持基于 CrossEncoder 的重排
- 支持文档上传、删除与知识库重建
- 提供离线评测脚本与测试集

## 技术栈

- Python
- FastAPI
- LangChain
- ChromaDB
- DashScope 兼容模型接口
- HuggingFace CrossEncoder
- Streamlit
- PyTorch
说明：模型是阿里云百炼的qwen系列模型，可替换为其他兼容模型。
## 项目结构

```text
RAG_project/
|- app/
|  |- main.py
|  |- config.py
|  |- streamlit_app.py
|  |- rag/
|  |  |- loader.py
|  |  |- splitter.py
|  |  |- vector_store.py
|  |  |- bm25_retriever.py
|  |  |- retriever.py
|  |  |- reranker.py
|  |  `- chain.py
|  `- models/
|- data/
|  |- raw/
|  `- chroma_db/
|- scripts/
|  `- ingest.py
|- eval/
|  |- rag_evaluate.py
|  `- test_dataset.json
|- requirements.txt
|- .gitignore
`- README.md
```

## 核心流程

```text
原始文档
  -> 文档加载
  -> 文本切分
  -> 向量化并写入 Chroma
  -> BM25 + 向量检索
  -> 重排
  -> 组装提示词
  -> 大模型生成回答
  -> 返回答案与引用来源
```

## 运行要求

- Python 3.10 及以上
- 可用的 DashScope 兼容聊天模型与向量模型
- 本地重排模型文件（如需启用重排，先执行 `python app/rag/download_model.py` 下载指定模型文件）

### 本地重排模型说明

如果需要启用重排（rerank），请先下载本地模型文件：

```bash
python app/rag/download_model.py
```

当前下载脚本会将模型默认保存到：

```text
C:/python-project/RAG_project/app/models/bge-reranker-v2-m3
```

下载完成后，请将 `.env` 中的 `RERANKER_MODEL_PATH2` 改为你本机实际模型路径。例如：

```env
RERANKER_MODEL_PATH2=C:/python-project/RAG_project/models/bge-reranker-v2-m3
```

如果你希望继续使用当前 README 示例中的路径，也可以将下载后的模型目录手动移动到：

```text
app/models/bge-reranker-v2-m3
```

## 安装依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 锁定的项目依赖版本如下：

```txt
fastapi==0.138.0
uvicorn==0.49.0
python-multipart==0.0.32

langchain-openai==1.3.2
langchain-community==0.4.2
langchain-chroma==1.1.0
langchain-classic==1.0.8
langchain-core==1.4.8
langchain-text-splitters==1.1.2

chromadb==1.5.9
sentence-transformers==5.6.0
transformers==5.12.1
torch==2.12.1
modelscope==1.37.1

pydantic==2.13.4
pydantic-settings==2.14.2
python-dotenv==1.2.2

unstructured==0.23.1
markdown==3.10.2
jieba==0.42.1
requests==2.34.2
streamlit==1.58.0
```

## 环境配置

在项目根目录创建 `.env`（注意文件名就叫.env） 文件，例如：

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen3.7-plus
EMBEDDING_MODEL=text-embedding-v4
CHROMA_DB_PATH=./data/chroma_db
RAW_DATA_PATH=./data/raw
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=3
RERANKER_MODEL_PATH2=C:/python-project/RAG_project/app/models/bge-reranker-v2-m3
```

相关配置项定义见 `app/config.py`。

## 最小可运行流程

第一次运行项目时，建议按下面顺序执行：

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 在项目根目录创建并填写 `.env`

3. 下载本地重排模型（如需启用重排）

```bash
python app/rag/download_model.py
```

4. 构建知识库向量索引

```bash
python scripts/ingest.py --reset
```

5. 启动 FastAPI 服务

```bash
uvicorn app.main:app --reload
```

6. 打开接口文档或健康检查页面确认服务已启动

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
```

## 准备数据

将知识库源文档放入：

```text
data/raw/
```

仓库中可以保留脱敏后的示例文档用于演示；本地索引文件建议重新构建，不建议直接纳入版本控制。
（如果不需要运行评测，可直接删除 data/raw/ 下的示例文件）
说明：数据来源为阿里云百炼知识库，已脱敏。
## 构建知识库

执行：

```bash
python scripts/ingest.py --reset
```

该步骤会完成以下工作：

- 读取 `data/raw/` 下的文档
- 切分为检索片段
- 重建 Chroma 向量库
- 将索引写入 `data/chroma_db/`

## 启动服务

执行：

```bash
uvicorn app.main:app --reload
```

启动后可访问：

- Swagger 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

## API 示例

### `POST /ask`

问答接口示例请求：

```json
{
  "question": "商品签收后还能申请退货吗？",
  "top_k": 3,
  "use_rerank": true,
  "rerank_top_n": 3
}
```

示例响应：

```json
{
  "answer": "根据召回到的知识库内容，若满足对应售后规则，则可以继续申请退货。",
  "sources": [
    {
      "id": 1,
      "source": "02_seven_day_no_reason.md",
      "content": "...",
      "metadata": {}
    }
  ]
}
```

### 其他接口

- `GET /documents`：查看当前原始文档列表
- `POST /documents/upload`：上传新的知识库文档
- `POST /knowledge/rebuild`：重建知识库索引
- `DELETE /documents/{filename}`：删除指定文档

## 评测

执行：

```bash
python eval/rag_evaluate.py
```
说明：评测脚本初始框架由AI辅助生成，只能运行评测文件，结合本项目RAG链路完成业务改造与指标扩充。
评测运行仍依赖 .env、向量库构建结果和本地重排模型
（只为测试原有data/raw文件测试，请勿用于生产环境）

- 评测脚本：`eval/rag_evaluate.py`
- 测试集：`eval/test_dataset.json`

## 数据与版本控制说明

建议公开仓库遵循以下策略：

- `data/raw/` 仅保留示例或已脱敏文档
- `data/chroma_db/` 不纳入版本控制
- `.env` 不上传
- 本地模型文件按需自行准备，不建议直接上传

## 项目限制

- 公开仓库仅保留脱敏后的示例知识库文档，不包含完整业务数据
- `data/chroma_db/` 不纳入版本控制，需要在本地通过 `scripts/ingest.py` 重新构建
- 本地重排模型文件需要单独下载，仓库中默认不提供
- 评测脚本目前是可运行的初始框架，主要用于演示 RAG 链路验证，不等同于完整评测平台
- 当前项目以本地开发和面试展示为目标，尚未补充 Docker 化部署、线上观测、权限控制和持续集成

## 下一步优化

- 增加更系统的评测指标，如命中率、MRR、重排前后对比和答案可用性分析
- 优化文本切分策略，结合售后规则文档结构调整 `chunk_size`、`chunk_overlap` 和 metadata
- 增强召回链路，尝试 BM25、向量检索、重排参数的系统调优，进一步提升复杂多诉求问题的命中率
- 为问答接口补充缓存、异常兜底和无答案返回策略，降低幻觉风险
- 补充 Docker、部署文档和最小监控方案，将项目从本地演示版本推进到可部署版本

## 补充说明

- 仓库中同时保留了 Streamlit 相关文件，但当前主要后端入口为 `app/main.py`
- 如果修改了知识库原始文档，建议重新执行 `scripts/ingest.py` 或调用重建接口更新索引

## License

本项目采用 MIT License，详见 [LICENSE](./LICENSE)。
