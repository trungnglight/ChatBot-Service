from fastapi import FastAPI, Body, UploadFile, status, HTTPException, Path
from langchain_core.messages import BaseMessage
from app.chatbot import ChatBot
from app.chroma import VectorDB
from pydantic import BaseModel
import uuid
from typing import Annotated

user_id_list = ["non_user"]
user_chat_list = {"non_user": ["94a10151-3fd6-443e-a3c8-31cc2b7ca988"]}
chat_list = ["94a10151-3fd6-443e-a3c8-31cc2b7ca988"]
chat_history = {"94a10151-3fd6-443e-a3c8-31cc2b7ca988": []}

app = FastAPI()
SYSTEM_MESSAGE = "Bạn là một trợ lý ảo hữu ích. Câu trả lời của bạn ngắn gọn và vào thẳng trọng tâm của vấn đề"

chatbot = ChatBot(SYSTEM_MESSAGE)
vectorDB = VectorDB("file_upload")
bot_id = "chatbot"


class User(BaseModel):
    user_id: str


class Chat(User):
    chat_id: str


class ChatMessage(Chat):
    message: str


class ChatHistory(Chat):
    history: list[BaseMessage] = []


@app.post("/chatbot/new_chat", status_code=status.HTTP_201_CREATED, response_model=Chat)
def create_new_chat(
    user_id: Annotated[str, Body(embed=True)],
):
    if user_id not in user_id_list:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    new_chat_id = str(uuid.uuid4())
    user_chat_list[user_id].append(new_chat_id)
    chat_list.append(new_chat_id)
    chat_history[new_chat_id] = []

    return Chat(user_id=user_id, chat_id=new_chat_id)


@app.post(
    "/chatbot/chat/{chat_id}",
    status_code=status.HTTP_200_OK,
    response_model=ChatMessage,
)
def get_response_for_chat(
    user_id: Annotated[str, Body(embed=True)],
    chat_id: Annotated[str, Path()],
    message: Annotated[str, Body(embed=True)],
):
    if (
        chat_id not in chat_list
        or chat_id not in chat_history.keys()
        or chat_id not in user_chat_list[user_id]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found"
        )
    history = chat_history[chat_id]
    response, new_history = chatbot.generate_answer(message, history)
    chat_history[chat_id] = new_history
    return ChatMessage(user_id=bot_id, chat_id=chat_id, message=response)


@app.get("/chatbot/chat/{chat_id}")
def get_chat(chat_id: Annotated[str, Path(embed=True)]):
    # history = randomDB.query(chat.chat_id)
    history = chat_history[chat_id]
    return {"chat_id": chat_id, "history": history}


@app.post("/upload_file")
def upload_file(uploaded_file: UploadFile):
    return {"file_type": uploaded_file.content_type}
    # vectorDB.add_file(uploaded_file.file, uploaded_file.content_type)
