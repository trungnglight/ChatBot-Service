import streamlit as st
from app.chatbot import ChatBot
from app.chroma import VectorDB
from io import BytesIO
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from typing import Literal, Iterator, Iterable

# Type for intent
ValidIntent = Literal["analyze_excel", "summarize_text", "chat"]

# Prompts
SYSTEM_MESSAGE = "Bạn là một trợ lý AI chuyên phân tích dữ liệu và tóm tắt nội dung từ các tệp Excel, văn bản, PDF và Word. Trả lời chính xác và rõ ràng theo yêu cầu của người dùng."
SUMMARY_MESSAGE = "Dựa trên câu trả lời sau, hãy tạo một tiêu đề ngắn gọn (tối đa 7 từ) để mô tả nội dung chính của đoạn hội thoại. Chỉ trả lời tiêu đề đó."


def chatbot_stream_wrapper(stream: Iterator):
    for message_chunk, _ in stream:
        if message_chunk.content:
            yield message_chunk.content


class ChatBotPage:
    # Constructor (Set default value)
    def __init__(_self):
        ss = st.session_state
        ss.setdefault("chat_messages", [])
        ss.setdefault("user_list", ["non_user"])
        ss.setdefault("user_chat_list", {"non_user": []})
        ss.setdefault("chat_list", {})
        ss.setdefault("chat_history", {})
        ss.setdefault("chat_send", False)
        ss.setdefault("delete_chat_id", None)
        ss.setdefault("intent", "chat")
        ss.setdefault("uploaded_files", {})
        ss.setdefault("chat_uploaded_files", {})

    # Create ChatBot, VectorDB and save to cache
    @st.cache_resource(ttl=6000, show_spinner="Đang khởi tạo ChatBot...")
    def init_model(_self):
        return ChatBot(SYSTEM_MESSAGE)

    @st.cache_resource(ttl=6000, show_spinner="Đang khởi tạo CSDL nội bộ...")
    def init_database(_self):
        return VectorDB("local_doc")

    # Main website function
    def load_chatbot(_self):
        # Logo
        st.logo(
            image=".\\images\\logo.png",
            size="large",
        )
        # Set current chat and create all cache object
        ss = st.session_state
        query_params = st.query_params
        current_chat_id = query_params.get("chat_id")
        chatbot = _self.init_model()
        database = _self.init_database()

        # Sidebar: User login
        st.sidebar.text_input(
            "Nhập tên người dùng",
            key="user_id",
            on_change=st.query_params.clear(),
        )
        if ss.user_id not in ss.user_list:
            ss.user_list.append(ss.user_id)
        ss.user_chat_list.setdefault(ss.user_id, [])

        # Sidebar: Upload documents
        with st.sidebar.form("add_documents", border=False):
            uploaded = st.file_uploader(
                "Upload file",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
            )
            add_btn = st.form_submit_button("Tải lên")
        if add_btn and uploaded:
            with st.spinner("Đang thêm dữ liệu..."):
                for f in uploaded:
                    database.add_file(f)
                    ss.uploaded_files[f.name] = f.getvalue()

        # Sidebar: Delete all data
        if st.sidebar.button("Xoá toàn bộ dữ liệu"):
            database.delete_all_data(ss.user_id)
            st.sidebar.success("Đã xoá.")

        # Sidebar: Chat list
        st.sidebar.title("Các cuộc nói chuyện")
        for id in ss.user_chat_list[ss.user_id]:
            col1, col2 = st.sidebar.columns([4, 1])
            title = ss.chat_list.get(id, "No Title")
            display_title = f"{title}"
            if col1.button(display_title, key=f"chatbtn_{id}"):
                st.query_params["chat_id"] = id
                st.rerun()
            if col2.button("🗑️", key=f"delete_{id}"):
                ss.delete_chat_id = id

        # Handle deletion
        if ss.delete_chat_id:
            id = ss.delete_chat_id
            if id in ss.user_chat_list[ss.user_id]:
                ss.user_chat_list[ss.user_id].remove(id)
            ss.chat_list.pop(id, None)
            ss.chat_history.pop(id, None)
            if query_params.get("chat_id") == id:
                st.query_params.clear()
            ss.delete_chat_id = None
            ss.chat_send = False
            st.rerun()

        # Sidebar: Create new chat (clears query param)
        if st.sidebar.button("Tạo cuộc trò chuyện mới"):
            st.query_params.clear()
            ss.chat_send = False
            st.rerun()

        # Main section
        st.title("Trò chuyện")

        # Intent selection box for easy changing
        st.selectbox(
            label="Intent",
            options=["chat", "analyze_excel", "summarize_text"],
            format_func=lambda x: {
                "chat": "💬 Trò chuyện thông thường",
                "analyze_excel": "📊 Phân tích tệp Excel",
                "summarize_text": "📝 Tóm tắt nội dung tài liệu",
            }[x],
            key="intent",
            # horizontal=False,
            label_visibility="collapsed",
            width=275,
        )

        # Input field
        st.chat_input(
            "...",
            key="message",
            accept_file=True,
            file_type=["xlsx", "docx", "pdf", "txt"],
            on_submit=lambda: (ss.update({"chat_send": True})),
        )

        # Trigger on chat_input message sent
        if ss.chat_send and ss.message:
            # Get uploaded file and put into a temporary list
            filedata = (
                ss.message.files[-1]
                if hasattr(ss.message, "files") and ss.message.files
                else None
            )
            if filedata:
                ss.uploaded_files["temp_input"] = filedata.getvalue()

            # New chat
            if not current_chat_id:

                # Create new id and add to storage
                new_id = str(uuid.uuid4())
                if ss.user_id not in ss.user_chat_list:
                    ss.user_chat_list[ss.user_id] = []
                ss.user_chat_list[ss.user_id].append(new_id)

                # Print out user message
                with st.chat_message("human"):
                    st.write(ss.message.text)

                # Generate AI message
                response = chatbot.generate_answer_stream(
                    user_input=ss.message.text,
                    chat_history=[],
                    file=(
                        BytesIO(ss.uploaded_files.get("temp_input"))
                        if "temp_input" in ss.uploaded_files
                        else None
                    ),
                    intent=ss.intent,
                )

                with st.chat_message("ai"):
                    full_response = st.write_stream(chatbot_stream_wrapper(response))

                # Write result to chat history
                ss.chat_history[new_id] = [
                    HumanMessage(content=ss.message.text),
                    AIMessage(content=full_response),
                ]

                # Create a summary as title for the chat and save it
                summary = chatbot.generate_answer_text(
                    user_input="Summarise this in 5 words. Return only the summary: "
                    + full_response,
                    chat_history=[],
                    file=None,
                    intent="chat",
                )
                ss.chat_list[new_id] = summary

                # Redirect to newly created chat
                st.query_params["chat_id"] = new_id
                st.rerun()

            # Existing chat
            else:
                # Show previous message
                if current_chat_id and current_chat_id in ss.chat_history:
                    for msg in ss.chat_history[current_chat_id]:
                        with st.chat_message("user" if msg.type == "human" else "ai"):
                            st.write(msg.content)
                with st.chat_message("human"):
                    st.write(ss.message.text)

                # Get chat history and generate AI message with it
                chat_hist = ss.chat_history.setdefault(current_chat_id, [])
                response = chatbot.generate_answer_stream(
                    user_input=ss.message.text,
                    chat_history=chat_hist,
                    file=(
                        BytesIO(ss.uploaded_files.get("temp_input"))
                        if "temp_input" in ss.uploaded_files
                        else None
                    ),
                    intent=ss.intent,
                )
                with st.chat_message("ai"):
                    full_response = st.write_stream(chatbot_stream_wrapper(response))

                # Add new messages to chat history
                ss.chat_history[current_chat_id] = chat_hist + [
                    HumanMessage(content=ss.message.text),
                    AIMessage(content=full_response),
                ]
                ss.chat_send = False
                st.rerun()

        # Print out all message after the rerun clear it out.
        if current_chat_id and current_chat_id in ss.chat_history:
            for msg in ss.chat_history[current_chat_id]:
                with st.chat_message("user" if msg.type == "human" else "ai"):
                    st.write(msg.content)


# Run the app
ChatBotPage().load_chatbot()
