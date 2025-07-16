import streamlit as st
from app.chatbot import ChatBot
from app.chroma import VectorDB
from langchain_core.messages import HumanMessage, AIMessage
from typing import Iterator
import uuid

# Prompts
SYSTEM_MESSAGE = "Bạn là một trợ lý ảo hữu ích. Câu trả lời của bạn ngắn gọn và vào thẳng trọng tâm của vấn đề."
SUMMARY_MESSAGE = "Dựa trên câu trả lời sau, hãy tạo một tiêu đề ngắn gọn (tối đa 7 từ) để mô tả nội dung chính của đoạn hội thoại. Chỉ trả lời tiêu đề đó."


def response_wrapper(response_stream: Iterator):
    for chunk in response_stream:
        yield chunk[0].content


class ChatBotPage:
    def __init__(_self):
        ss = st.session_state
        ss.setdefault("chat_messages", [])
        ss.setdefault("user_list", ["non_user"])
        ss.setdefault("user_chat_list", {"non_user": []})
        ss.setdefault("chat_list", {})
        ss.setdefault("chat_history", {})
        ss.setdefault("chat_send", False)
        ss.setdefault("delete_chat_id", None)

    @st.cache_resource(ttl=6000, show_spinner="Đang khởi tạo ChatBot...")
    def init_model(_self):
        return ChatBot(SYSTEM_MESSAGE)

    @st.cache_resource(ttl=6000, show_spinner="Đang khởi tạo CSDL nội bộ...")
    def init_database(_self):
        return VectorDB("local_doc")

    def load_chatbot(_self):
        ss = st.session_state
        query_params = st.query_params
        current_chat_id = query_params.get("chat_id")
        chatbot = _self.init_model()
        database = _self.init_database()

        # Sidebar: User login
        st.sidebar.text_input("Nhập tên người dùng", key="user_id", on_change=l)
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

        # Sidebar: Delete all data
        if st.sidebar.button("Xoá toàn bộ dữ liệu"):
            database.delete_all_data(ss.user_id)
            st.sidebar.success("Đã xoá.")

        # Sidebar: Chat list
        st.sidebar.title("Các cuộc nói chuyện")
        for id in ss.user_chat_list[ss.user_id]:
            col1, col2 = st.sidebar.columns([4, 1])
            title = ss.chat_list.get(id, "No Title")
            if col1.button(title, key=f"chatbtn_{id}"):
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
            st.rerun()

        # Sidebar: Create new chat (clears query param)
        if st.sidebar.button("Tạo cuộc trò chuyện mới"):
            st.query_params.clear()
            st.rerun()

        # Main section
        st.header("Trò chuyện")
        st.chat_input(
            "...", key="message", on_submit=lambda: ss.update({"chat_send": True})
        )

        if ss.chat_send and ss.message:
            if not current_chat_id:
                # New chat is created now
                new_id = str(uuid.uuid4())
                if ss.user_id not in ss.user_chat_list:
                    ss.user_chat_list[ss.user_id] = []
                ss.user_chat_list[ss.user_id].append(new_id)

                # Write current message
                with st.chat_message("human"):
                    st.write(ss.message)

                # Step 1: Generate AI response
                response = chatbot.generate_answer_text(ss.message)
                with st.chat_message("ai"):
                    st.write(response)
                ss.chat_history[new_id] = [
                    HumanMessage(content=ss.message),
                    AIMessage(content=response),
                ]

                # Step 2: Generate short title from response
                summary_bot = ChatBot(SUMMARY_MESSAGE)
                summary = summary_bot.generate_answer_text(response)

                # Step 3: Save to session
                ss.chat_list[new_id] = summary
                st.query_params["chat_id"] = new_id
                st.rerun()
            else:
                if current_chat_id and current_chat_id in ss.chat_history:
                    for msg in ss.chat_history[current_chat_id]:
                        with st.chat_message("user" if msg.type == "human" else "ai"):
                            st.write_stream(msg.content)
                with st.chat_message("human"):
                    st.write(ss.message)
                chat_hist = ss.chat_history.setdefault(current_chat_id, [])
                response = chatbot.generate_answer_text(ss.message, chat_hist)
                with st.chat_message("ai"):
                    st.write(response)
                ss.chat_history[current_chat_id] = chat_hist + [
                    HumanMessage(content=ss.message),
                    AIMessage(content=response),
                ]
                ss.chat_send = False
                st.rerun()

        # Display chat
        if current_chat_id and current_chat_id in ss.chat_history:
            for msg in ss.chat_history[current_chat_id]:
                with st.chat_message("user" if msg.type == "human" else "ai"):
                    st.write(msg.content)


# Run the app
ChatBotPage().load_chatbot()
