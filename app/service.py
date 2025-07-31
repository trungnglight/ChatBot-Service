from fastapi import FastAPI, Body, UploadFile, status
from langchain_core.messages import BaseMessage
from app.chatbot import ChatBot
from app.chroma import VectorDB
from pydantic import BaseModel
from typing import Annotated, Optional, Iterator


user_id_list = ["non_user"]
user_chat_list = {"non_user": ["94a10151-3fd6-443e-a3c8-31cc2b7ca988"]}
chat_list = ["94a10151-3fd6-443e-a3c8-31cc2b7ca988"]
chat_history = {"94a10151-3fd6-443e-a3c8-31cc2b7ca988": []}

app = FastAPI()
SYSTEM_MESSAGE = "Bạn là một trợ lý AI chuyên phân tích dữ liệu và tóm tắt nội dung từ các tệp Excel, văn bản, PDF và Word. Trả lời chính xác và rõ ràng theo yêu cầu của người dùng."

vectorDB = VectorDB("docling_file")
chatbot = ChatBot(SYSTEM_MESSAGE)
bot_id = "chatbot"


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    message: Iterator


class ChatHistory(BaseModel):
    history: list[BaseMessage] = []


# @app.post("/chatbot/new_chat", status_code=status.HTTP_201_CREATED, response_model=Chat)
# def create_new_chat(
#     user_id: Annotated[str, Body(embed=True)],
# ):
#     if user_id not in user_id_list:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
#         )
#     new_chat_id = str(uuid.uuid4())
#     user_chat_list[user_id].append(new_chat_id)
#     chat_list.append(new_chat_id)
#     chat_history[new_chat_id] = []

#     return Chat(user_id=user_id, chat_id=new_chat_id)


@app.post(
    "/chatbot/chat",
    status_code=status.HTTP_200_OK,
    response_model=ChatResponse,
)
def get_response_for_chat(
    message: Annotated[str, Body(embed=True)],
    file: Optional[UploadFile] = None,
    intent: Annotated[Optional[str], Body(embed=True)] = "chat",
    chat_history: Annotated[list[BaseMessage], Body(embed=True)] = [],
):

    if file:
        response = chatbot.generate_answer_stream(
            message,
            chat_history,
            file=file.file,
            content_type=file.content_type,
            intent=intent,
        )
    else:
        response = chatbot.generate_answer_stream(message, chat_history, intent="chat")
    return ChatResponse(message=response)


# @app.get("/chatbot/chat/{chat_id}")
# def get_chat(chat_id: Annotated[str, Path(embed=True)]):
#     # history = randomDB.query(chat.chat_id)
#     history = chat_history[chat_id]
#     return {"chat_id": chat_id, "history": history}


@app.post("/upload_file", status_code=status.HTTP_200_OK)
def upload_file(uploaded_file: UploadFile):
    text_chunk = vectorDB.add_file(
        uploaded_file.filename,
        uploaded_file.file,
        uploaded_file.content_type,
    )
    return {"text_chunk": text_chunk}
