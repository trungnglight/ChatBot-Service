import os
import requests
import pandas as pd
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from io import BytesIO

# === ENV SETUP ===
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LL_MODEL = os.getenv("LL_MODEL", "llama3.2:3b")

# === PROMPT ===
PROMPT = PromptTemplate.from_template(
    """
{system_message}

Lịch sử nói chuyện:
{chat_history}

Câu hỏi người dùng:
{question}

Answer:
"""
)


# === TOOL: Excel Auto-Cleanup ===
class ExcelAnalysisInput(BaseModel):
    file: Annotated[BytesIO, Field(description="Excel file")]
    instruction: Annotated[
        str, Field(description="User instruction or analysis request")
    ]
    model_config = {"arbitrary_types_allowed": True}


def auto_clean_excel(file: BytesIO) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, thresh=len(df) * 0.5)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].mean())
    df = df.drop_duplicates()
    return df


def analyze_excel_with_cleanup(file: BytesIO, instruction: str) -> str:
    try:
        df = auto_clean_excel(file)

        # Use pandas agent to dynamically interpret instructions
        agent = create_pandas_dataframe_agent(  # Experimental only, do not use
            llm=ChatOpenAI(
                base_url=f"{OLLAMA_HOST}/v1/",
                model=LL_MODEL,
                api_key="ollama",
                temperature=0,
            ),
            df=df,
            verbose=False,
            allow_dangerous_code=True,
        )

        return agent.invoke(instruction)

    except Exception as e:
        return f"Error processing Excel file: {str(e)}"


excel_tool = StructuredTool.from_function(
    name="AnalyzeExcelWithInstruction",
    func=analyze_excel_with_cleanup,
    description="Auto-cleans and analyzes an Excel file. Provide file and instruction.",
    args_schema=ExcelAnalysisInput,
    return_direct=True,
)


# === INTENT PARSER ===
class ExcelIntent(BaseModel):
    intent: str = Field(description="One of: 'analyze_excel', 'chat'")
    instruction: str | None = None


llm_for_intent = ChatOpenAI(
    base_url=f"{OLLAMA_HOST}/v1/",
    model=LL_MODEL,
    api_key="ollama",
    temperature=0,
)


intent_parser = PydanticOutputParser(pydantic_object=ExcelIntent)
intent_parser = OutputFixingParser.from_llm(parser=intent_parser, llm=llm_for_intent)
intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI that classifies if a user wants to analyze an Excel file.\n"
            "Return only JSON in the following format, do not explain anything:\n"
            "{{\n"
            '  "intent": "analyze_excel" | "chat",\n'
            '  "instruction": "..."\n'
            "}}",
        ),
        ("user", "Input: {input}"),
    ]
)
intent_chain = intent_prompt | llm_for_intent | intent_parser


# === STATE TYPE ===
class State(TypedDict):
    input: str
    response: str
    chat_history: list[BaseMessage]
    file: BytesIO | None


# === CHATBOT CLASS ===
class ChatBot:
    def __init__(self, system_message: str) -> None:
        self.system_message = system_message

        requests.post(
            f"{OLLAMA_HOST}/api/pull",
            json={"model": LL_MODEL},
        )

        self.llm = ChatOpenAI(
            base_url=f"{OLLAMA_HOST}/v1/",
            model=LL_MODEL,
            api_key="ollama",
            temperature=0.4,
            max_completion_tokens=800,
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

    def detect_excel_request(self, message: str) -> ExcelIntent:
        try:
            return intent_chain.invoke({"input": message})
        except Exception as e:
            print(f"[Intent detection error] {e}")
            print(
                "[Hint] The model likely returned malformed JSON. Check prompt formatting or try OutputFixingParser."
            )
            return ExcelIntent(intent="chat")

    def generate_node(self, state: State) -> State:
        chat_str = self.format_chat_history(state["chat_history"])
        question = state["input"]
        file = state["file"]

        intent_result = self.detect_excel_request(question)

        if intent_result.intent == "analyze_excel" and file:
            try:
                tool_output = excel_tool.run(
                    {
                        "file": file,
                        "instruction": intent_result.instruction or "summarize",
                    }
                )
                response = f"[Excel Analysis Result]\n{tool_output}"
            except Exception as e:
                response = f"[Error running Excel tool] {e}"
        else:
            response_obj = self.llm_chain.invoke(
                {
                    "system_message": self.system_message,
                    "question": question,
                    "chat_history": chat_str,
                }
            )
            response = (
                response_obj.content
                if hasattr(response_obj, "content")
                else str(response_obj)
            )

        return {**state, "response": response}

    def generate_answer(
        self,
        user_input: str,
        chat_history: list[BaseMessage] = [],
        file: BytesIO | None = None,
    ) -> tuple[str, list[BaseMessage]]:
        state: State = {
            "input": user_input,
            "chat_history": chat_history,
            "response": "",
            "file": file,
        }
        result = self.graph.invoke(state)
        new_history = chat_history + [
            HumanMessage(content=user_input),
            AIMessage(content=result["response"]),
        ]
        return result["response"], new_history


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    bot = ChatBot("Bạn là một trợ lý phân tích dữ liệu chuyên đọc file Excel.")
    history: list[BaseMessage] = []

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        response, history = bot.generate_answer(user_input, history)
        print("Bot:", response)
