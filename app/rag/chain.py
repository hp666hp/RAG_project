from typing import Any
from langchain_core.documents import Document
from app.rag.reranker import retriever_top_k
from app.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 无法回答时的默认回复
NO_ANSWER = "抱歉，根据提供的上下文信息，我无法回答这个问题。"

# 系统提示词
SYSTEM_PROMPT = """
你是一个智能问答助手。请根据提供的<context>上下文信息来回答用户的问题。

回答规则：
1. 严格基于 <context> 中的信息进行回答，不要编造事实。
2. 如果 <context> 中没有包含回答问题所需的信息，请直接回答：“抱歉，根据提供的上下文信息，我无法回答这个问题。”
3. 在回答中引用来源时，请使用脚注格式，例如 [1]、[2] 等。
4. 保持回答简洁、准确、专业。
5. 使用中文进行回答。
""".strip()

# 用户提示词模板
USER_PROMPT = """
<context>
{context}
</context>

<question>
{question}
</question>
""".strip()


def create_chat_model():
    """
    创建并返回 ChatOpenAI 模型实例
    
    Returns:
        ChatOpenAI: 配置好的聊天模型实例
    """
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0,  # 设置为0以获得确定性输出
        streaming=True,
    )


def format_docs(docs: list[Document]) -> str:
    """
    将检索到的文档片段格式化为上下文字符串
    """
    context_parts = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"[{index}] 来源: {source}\n内容: {doc.page_content}"
        )
    
    return "\n\n".join(context_parts)

def build_context(question: str, top_k: int = 3) -> str:
    """检索并构造上下文。"""
    docs = retriever_top_k(
        question=question,
        top_k=top_k,
        use_rerank=True,
        rerank_top_n=3,
    )
    context = format_docs(docs)
    return context

def extract_sources(docs: list[Document]) -> list[dict]:
    """提取检索结果来源信息。"""

    sources = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        sources.append(
            {
                "id": index,
                "source": metadata.get("source", "unknown"),
                "content": doc.page_content,
                "metadata": metadata,
            }
        )

    return sources


def answer_question(
    question: str,
    top_k: int=3,
    use_rerank: bool = True,
    rerank_top_n: int = 3,
) -> dict[str, Any]:

    """
    回答用户提出的问题
    
    Args:
        use_rerank:
        rerank_top_n:
        question (str): 用户的问题
        top_k (int | None): 检索的文档片段数量，默认为配置中的值
        
    Returns:
        dict[str, Any]: 包含 'answer' (回答内容) 和 'sources' (来源列表) 的字典
        
    Raises:
        ValueError: 当问题为空时抛出异常
    """
    question_clean = question.strip()
    if not question_clean:
        raise ValueError("问题不能为空")

    # 检索相关文档
    retrieved_chunks = retriever_top_k(question_clean, top_k, use_rerank, rerank_top_n)

    # 如果没有检索到相关文档，直接返回默认回复
    if not retrieved_chunks:
        return {
            "answer": NO_ANSWER,
            "sources": [],
        }

    # 格式化上下文
    context = format_docs(retrieved_chunks)

    # 构建提示词
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )

    # 构建处理链：提示词 -> 模型 -> 字符串解析器
    chain = prompt | create_chat_model() | StrOutputParser()
    
    # 执行链并获取答案
    answer = chain.invoke({
        "context": context,
        "question": question_clean,
    })

    return {
        "answer": answer,
        "sources": extract_sources(retrieved_chunks),
    }


