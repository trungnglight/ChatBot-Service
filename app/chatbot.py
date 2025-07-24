import os
import requests
import pandas as pd
from io import BytesIO
from tempfile import SpooledTemporaryFile
from typing import TypedDict, BinaryIO
from enum import Enum
from pydantic import BaseModel, SecretStr
import magic

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LL_MODEL = os.getenv("LL_MODEL", "llama3.2:3b")
OLLAMA_API_KEY = SecretStr("ollama")

PROMPT = PromptTemplate.from_template(
    """
{system_message}
Nếu câu hỏi người dùng trống, yêu cầu người dùng cung cấp câu hỏi.

Lịch sử nói chuyện:
{chat_history}

Câu hỏi người dùng:
{question}

Answer:
"""
)


class IntentEnum(str, Enum):
    analyze_excel = "analyze_excel"
    summarize_text = "summarize_text"
    chat = "chat"


class FileIntent(BaseModel):
    intent: IntentEnum
    instruction: str | None = None


intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intelligent assistant specialized in detecting user intent related to file analysis and summarization tasks.

Analyze the input carefully and return a JSON object with:
- 'intent': one of ['analyze_excel', 'summarize_text', 'chat']
- 'instruction': a clear and specific instruction the assistant should follow when processing the file (optional for 'chat' intent).

Examples:
- For Excel analysis: 'intent': 'analyze_excel', 'instruction': 'Get the average age of participants. Get the value with the highest percentage in each column'
- For summarization: 'intent': 'summarize_text', 'instruction': 'Summarize the document in Vietnamese.'
- For general chat: 'intent': 'chat'.

Only return JSON.
""",
        ),
        ("user", "Input: {input}"),
    ]
)

intent_llm = ChatOpenAI(
    base_url=f"{OLLAMA_HOST}/v1/",
    model=LL_MODEL,
    api_key=OLLAMA_API_KEY,
    temperature=0,
)

intent_parser = PydanticOutputParser(pydantic_object=FileIntent)
intent_parser = OutputFixingParser.from_llm(parser=intent_parser, llm=intent_llm)
intent_chain = intent_prompt | intent_llm | intent_parser


def agent_executor(agent, tools):
    return AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True
    )


def auto_clean_excel(file_stream: BinaryIO) -> pd.DataFrame:
    file_stream.seek(0)
    df = pd.read_excel(file_stream)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())
    df = df.drop_duplicates()
    return df


def analyze_dataframe_with_agent(df: pd.DataFrame, instruction: str) -> str:
    agent = create_pandas_dataframe_agent(
        llm=ChatOpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            model=LL_MODEL,
            api_key=OLLAMA_API_KEY,
            temperature=0,
        ),
        df=df,
        verbose=False,
        allow_dangerous_code=True,
    )
    safe_instruction = (
        """You are a Python data analyst with direct access to a runtime environment and can use pandas, numpy, and any built-in libraries to analyze Excel files.

        You have been provided with a DataFrame called df (or dfs if multiple sheets). Do not explain how to analyze Excel files — perform the analysis directly using Python code and return summaries or results.

        Your job is to:

            Understand the structure and key variables of the data. 

            Execute appropriate Python code using df to summarize and analyze the content.

            When asked a question or given a task, do not explain the method—just execute the logic and return the result.

            If the task is ambiguous, suggest the most likely interpretation, and proceed with it.

            Avoid tutorials or generic guidance—focus only on solving with data."""
        f"Instruction: {instruction.strip()}\n\n"
        "Only use the tools provided. Do not print or describe the dataframe directly."
    )
    analysis = agent.invoke(safe_instruction)
    return analysis if isinstance(analysis, str) else analysis["output"]


def analyze_excel_with_cleanup(file_stream: BinaryIO, instruction: str) -> str:
    try:
        df = auto_clean_excel(file_stream)
        return analyze_dataframe_with_agent(df, instruction)
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"


def summarize_with_docling(file: BinaryIO, content_type: str, instruction: str) -> str:
    try:
        name = {
            "application/pdf": "upload.pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "upload.docx",
            "text/plain": "upload.txt",
        }.get(content_type, "upload.unknown")

        file.seek(0)
        stream = BytesIO(file.read())
        markdown = ""
        if content_type == "text/plain":
            markdown = stream.getvalue().decode("utf-8")
        else:
            converter = DocumentConverter()
            doc = converter.convert(DocumentStream(name=name, stream=stream))
            markdown = doc.document.export_to_markdown()

        summarizer = ChatOpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            model=LL_MODEL,
            api_key=OLLAMA_API_KEY,
            temperature=0.2,
        )

        return summarizer.invoke(f"{instruction}\n\n{markdown[:5000]}").content
    except Exception as e:
        return f"[Docling extraction failed] {str(e)}"


class State(TypedDict):
    input: str
    response: str
    chat_history: list[BaseMessage]
    file: SpooledTemporaryFile | BytesIO | None
    content_type: str | None
    intent: IntentEnum | None


class ChatBot:
    def __init__(self, system_message: str) -> None:
        self.system_message = system_message
        requests.post(f"{OLLAMA_HOST}/api/pull", json={"model": LL_MODEL})

        self.llm = ChatOpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            model=LL_MODEL,
            api_key=OLLAMA_API_KEY,
            temperature=0.4,
            max_completion_tokens=2000,
            top_p=0.9,
        )
        self.llm_chain = PROMPT | self.llm

        graph_builder = StateGraph(State)
        graph_builder.add_node("generate", self.generate_node)
        graph_builder.add_edge(START, "generate")
        graph_builder.add_edge("generate", END)
        self.graph = graph_builder.compile()

    def format_chat_history(self, history: list[BaseMessage]) -> str:
        return "\n\n".join(
            f"{'User' if m.type == 'human' else 'AI'}: {m.content}" for m in history
        )

    def detect_intent(self, message: str) -> FileIntent:
        try:
            return intent_chain.invoke({"input": message})
        except Exception as e:
            print(f"[Intent detection error] {e}")
            return FileIntent(intent=IntentEnum.chat)

    def generate_node(self, state: State) -> State:
        chat_str = self.format_chat_history(state["chat_history"])
        user_input = state["input"]
        file = state["file"]
        content_type = state["content_type"]
        print("content_type: " + str(content_type))
        if content_type is None and file:
            try:
                file.seek(0)
                content_type = magic.from_buffer(file.read(2048), mime=True)
                print(content_type)
                file.seek(0)
            except Exception:
                content_type = "application/octet-stream"

        if isinstance(file, SpooledTemporaryFile):
            file = BytesIO(file.read())

        intent = FileIntent(intent=IntentEnum.chat)
        if state["intent"] is not None:
            intent.intent = state["intent"]
        else:
            if user_input:
                intent = self.detect_intent(user_input)

        if file and content_type:
            if intent.intent == IntentEnum.analyze_excel:
                print("Excel!")
                response = analyze_excel_with_cleanup(
                    file,
                    intent.instruction
                    or "Analyze- Summarize the sheet names and dimensions."
                    "- For each sheet: describe columns, types, missing values, and show basic stats."
                    "- Provide insights and suggestions for next steps.",
                )
                response = f"[Excel Analysis Result]\n{response}"
            elif intent.intent == IntentEnum.summarize_text:
                print("Summarize!")
                response = summarize_with_docling(
                    file, content_type, intent.instruction or "Tóm tắt nội dung văn bản"
                )
            elif intent.intent == IntentEnum.chat:
                print("Chat!")
                if user_input:
                    intent = self.detect_intent(user_input)
                    if intent.intent == IntentEnum.analyze_excel:
                        response = analyze_excel_with_cleanup(
                            file, intent.instruction or "analyze"
                        )
                        response = f"[Excel Analysis Result]\n{response}"
                    elif intent.intent == IntentEnum.summarize_text:
                        response = summarize_with_docling(
                            file,
                            content_type,
                            intent.instruction or "Tóm tắt nội dung văn bản",
                        )
                else:
                    llm_output = self.llm_chain.invoke(
                        {
                            "system_message": self.system_message,
                            "question": user_input,
                            "chat_history": chat_str,
                        }
                    )

                    response = (
                        llm_output.content
                        if hasattr(llm_output, "content")
                        else str(llm_output)
                    )

                    response = response if isinstance(response, str) else response[0]

        elif intent.intent == IntentEnum.analyze_excel:
            response = f"[Error] No file was uploaded, but intent was {intent.intent}"
        elif intent.intent == IntentEnum.summarize_text:
            llm_output = self.llm_chain.invoke(
                {
                    "system_message": self.system_message,
                    "question": "Summarise this:" + user_input,
                    "chat_history": chat_str,
                }
            )

            response = (
                llm_output.content
                if hasattr(llm_output, "content")
                else str(llm_output)
            )

            response = response if isinstance(response, str) else response[0]
        else:
            llm_output = self.llm_chain.invoke(
                {
                    "system_message": self.system_message,
                    "question": user_input,
                    "chat_history": chat_str,
                }
            )
            response = (
                llm_output.content
                if hasattr(llm_output, "content")
                else str(llm_output)
            )

            response = response if isinstance(response, str) else response[0]

        return {**state, "response": response}

    def generate_answer(
        self,
        user_input: str,
        chat_history: list[BaseMessage],
        file: SpooledTemporaryFile | BytesIO | None = None,
        content_type: str | None = None,
        intent: IntentEnum | None = None,
    ) -> str:
        state: State = {
            "input": user_input,
            "chat_history": chat_history,
            "response": "",
            "file": file,
            "content_type": content_type,
            "intent": intent,
        }
        result = self.graph.invoke(state)
        return result["response"]


if __name__ == "__main__":
    import mimetypes

    bot = ChatBot(
        "Bạn là một trợ lý AI chuyên phân tích dữ liệu và tóm tắt nội dung từ các tệp Excel, văn bản, PDF và Word. Trả lời chính xác và rõ ràng theo yêu cầu của người dùng."
    )
    history: list[BaseMessage] = []

    while True:
        mode = input("Enter 'chat' or 'file': ").strip().lower()
        if mode in ["exit", "quit"]:
            break

        if mode == "file":
            file_path = input("Enter file path: ").strip()
            instruction = input("Enter instruction: ").strip()
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = "application/octet-stream"
            with open(file_path, "rb") as f:
                file = SpooledTemporaryFile()
                file.write(f.read())
                file.seek(0)
                response = bot.generate_answer(
                    instruction, history, file=file, content_type=content_type
                )
                history += [
                    HumanMessage(content=instruction),
                    AIMessage(content=response),
                ]
                print("Bot:", response)
        else:
            user_text = input("User: ")
            response = bot.generate_answer(user_text, history)
            history += [
                HumanMessage(content=user_text),
                AIMessage(content=response),
            ]
            print("Bot:", response)
