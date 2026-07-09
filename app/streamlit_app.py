import requests
import streamlit as st
import time



# FastAPI 服务地址
API_BASE_URL = "http://127.0.0.1:8000"

# 配置 Streamlit 页面
st.set_page_config(page_title="RAG 智能问答系统", page_icon="🤖", layout="wide")
st.title("🤖 RAG 智能问答系统")


def api_get_documents():
    """获取已上传的文档列表"""
    response = requests.get(f"{API_BASE_URL}/documents", timeout=30)
    response.raise_for_status()
    return response.json()


def api_upload_file(uploaded_file):
    """上传文件到后端"""
    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files={
            "uploaded_file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def api_rebuild_knowledge():
    """重建知识库索引"""
    response = requests.post(f"{API_BASE_URL}/knowledge/rebuild", timeout=180)
    response.raise_for_status()
    return response.json()


def api_delete_document(file_name: str):
    """删除指定文档"""
    response = requests.delete(
        f"{API_BASE_URL}/documents/{file_name}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def api_ask(question: str, top_k: int,use_rerank: bool = True, rerank_top_n: int|None = None):
    """发送问题并获取回答"""
    response = requests.post(
        f"{API_BASE_URL}/ask",
        json={
            "question": question,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "rerank_top_n": rerank_top_n,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def build_multi_turn_question(question: str, turn_limit: int = 3) -> str:
    """
    构建包含历史对话上下文的 prompt
    :param question: 当前用户的问题
    :param turn_limit: 保留的历史对话轮数
    :return: 拼接好的 prompt 字符串
    """
    # 获取最近 turn_limit * 2 条消息（因为一问一答算两轮）
    history = st.session_state.messages[-turn_limit * 2 :]

    if not history:
        return question

    lines = []
    for message in history:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{role}: {message['content']}")

    return "以下是历史对话记录：\n" + "\n".join(lines) + f"\n\n当前问题: {question}"


# 初始化 session_state 用于存储聊天记录和状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "status" not in st.session_state:
    st.session_state.status = ""


# --- 侧边栏设置 ---
with st.sidebar:
    st.subheader("⚙️ 系统设置")

    # Top-K 检索数量设置
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=3)
    # 历史对话轮数限制
    turn_limit = st.slider("历史对话轮数", min_value=1, max_value=6, value=3)
    #重排序
    use_rerank = st.checkbox("开启重排序", value=True)
    rerank_top_n = 3
    if use_rerank:
        rerank_top_n = st.slider("重排保留数量", min_value=1, max_value=10, value=3)
        rerank_top_n = min(rerank_top_n, top_k)
    # 清空聊天历史按钮
    if st.button("🗑️ 清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # 显示操作状态信息
    if st.session_state.status:
        st.info(st.session_state.status)


    # --- 文件上传区域 ---
    st.subheader("📤 上传文档")

    uploaded_files = st.file_uploader(
        "支持 .txt 或 .md 格式",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("🚀 上传文件", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择文件！")
        else:
            try:
                uploaded_count = 0
                # 遍历上传的文件并发送到后端
                for uploaded_file in uploaded_files:
                    api_upload_file(uploaded_file)
                    uploaded_count += 1

                st.session_state.status = f"成功上传 {uploaded_count} 个文件，请点击重建索引以生效。"
                st.rerun()
            except Exception as exc:
                st.session_state.status = f"上传失败: {exc}"
                st.rerun()


    # --- 重建索引按钮 ---
    if st.button("🔄 重建知识库索引", use_container_width=True):
        try:
            result = api_rebuild_knowledge()
            st.session_state.status = (
                f"{result['message']} "
                f"共处理 {result['document_count']} 个文档, "
                f"{result['chunk_count']} 个 chunks"
            )
            st.rerun()
        except Exception as exc:
            st.session_state.status = f"重建索引失败: {exc}"
            st.rerun()


    # --- 文档管理列表 ---
    st.subheader("📄 文档管理")

    try:
        document_result = api_get_documents()
        documents = document_result.get("documents", [])
    except Exception as exc:
        st.error(f"获取文档列表失败: {exc}")
        documents = []

    if not documents:
        st.write("暂无已上传文档")
    else:
        for index, document in enumerate(documents):
            col1, col2 = st.columns([6, 1])

            with col1:
                st.write(document["file_name"])

            with col2:
                # 删除按钮
                if st.button("❌", key=f"delete_{index}"):
                    try:
                        api_delete_document(document["file_name"])
                        st.session_state.status = (
                            f"{document['file_name']} 已删除，请重新重建索引。"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.session_state.status = f"删除失败: {exc}"
                        st.rerun()


# --- 主界面聊天区域 ---
st.divider()
st.subheader("💬 智能问答")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # 如果是助手的回答且包含来源，则显示来源详情
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 查看参考来源"):
                for source in message["sources"]:
                    st.write(f"📄 来源: {source.get('source', 'unknown')}")
                    if source.get("content"):
                        st.write(source["content"])


# --- 用户输入处理逻辑 ---
question = st.chat_input("请输入问题")

if question:
    # 1. 将用户问题添加到历史记录
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # 2. 在界面上显示用户问题
    with st.chat_message("user"):
        st.markdown(question)

    # 3. 生成并显示助手回答
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_answer = ""
        sources = []
        with st.spinner("正在思考中..."):
            try:
                # 构建包含上下文的 prompt
                final_question = build_multi_turn_question(question, turn_limit)
                # 调用后端 API 获取回答

                result = api_ask(final_question, top_k, use_rerank, rerank_top_n)

                answer = result.get("answer", "抱歉，根据提供的上下文信息，我无法回答这个问题。")

                sources = result.get("sources", [])
                
                # 模拟打字机效果输出答案
                for char in answer:
                    full_answer += char
                    answer_placeholder.markdown(full_answer)
                    time.sleep(0.02)

                # 显示参考来源
                if sources  :
                    with sources_placeholder.expander("📚 查看参考来源"):
                        for source in sources:
                            st.write(f"📄 [{source['id']}] {source['source']}")
                            st.write(source["content"])

            except Exception as e:
                full_answer = f"发生错误: {e}"
                answer_placeholder.markdown(full_answer)


    # 4. 将助手的完整回答和来源存入历史记录
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        }
    )
