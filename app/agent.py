import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,Tool, AgentType
# from langchain_core.tools import tool

from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.google_serper.tool import GoogleSerperResults
from google.cloud import aiplatform
from google.oauth2 import service_account

from langchain_experimental.tools.python.tool import PythonREPLTool
import torch

load_dotenv()

credentials = service_account.Credentials.from_service_account_file(
    "credential.json"
)

# google_credentials_path = os.getenv("GOOGLE_CREDENTIAL")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
# Now your app should authenticate using the service account key.
# aiplatform.init(project="devops-440402", location="us-central1")


def is_greeting(query: str) -> bool:
    greetings = ["hi", "hello", "hey", "hai", "hola", "yo"]
    return query.strip().lower() in greetings

real_search_tool = GoogleSerperResults()

def safe_search(query: str) -> str:
    if is_greeting(query):
        return "Skipping search for greetings 😊"
    return real_search_tool.run(query)

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
tools = [
    Tool(
        name="SerpGoogleer Search",
        func=safe_search,
        description="Useful for answering questions about current events or recent info on the web",
        return_messages=True
    ),
    Tool.from_function(
        func=PythonREPLTool().run,
        name="Python REPL",
        description="Executes Python code"
    )
]


SYSTEM_PROMPT = """
You are a helpful AI agent that uses tools when needed.
Only do ONE of the following:

If you need to use a tool:
Thought: I need to use a tool
Action: <tool name>
Action Input: <input>

Otherwise:
Thought: I now know the final answer
Final Answer: <your answer>

Never output both an Action and Final Answer at the same time.
"""

# ✅ Dictionary untuk simpan memory per session
session_memories = {}




def create_agent(session_id: str):
    

    # open router
    # llm = ChatOpenAI(
    #     model="google/gemini-2.0-flash-exp:free",  # model dari OpenRouter
    #     openai_api_base=os.getenv("OPENAI_API_BASE"),
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     temperature=0.7,
    # )

    # together
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
        # model="meta-llama/Llama-4-Scout-17B-16E-Instruct", # model dari Together ai
        max_tokens=1024,
        openai_api_base="https://api.together.xyz/v1",
        openai_api_key="12180e976ee41b0a7777732964c45a6768a4d7678c45417df37e6ba042a1bf48",
        temperature=0.7,
    )

    # llm = VertexAI(
    #     model_name="gemma-3-12b-it",
    #     project="devops-440402",
    #     location="us-central1",
    #     temperature=0.7,
    #     credentials=credentials,
    #     verbose=True,
    #     max_output_tokens=1024  # ✅ This is the correct parameter
    # )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    message_history = RedisChatMessageHistory(
        url=redis_url,
        session_id=f"chat:{session_id}",
    )

    # ✅ Inject system message (Persona) hanya jika belum ada chat
    if not message_history.messages:
        system_message = SystemMessage(
            content=(
                "I am a AI Senior Brand Strategy Manager, specialize in brand positioning, market analysis, storytelling, and communication strategy. Respond with strategic insights, market context, and persuasive branding advice."
            )
        )
        message_history.add_message(system_message)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=message_history,
        k=5
    )

    
    # memory.chat_memory.add_message(system_message)  # Inject persona to memory

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        # agent=AgentType.OPENAI_FUNCTIONS, # ⚠️ Bukan "MULTI_FUNCTIONS", karena Together belum full support
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=1,
        handle_parsing_errors=True,
        early_stopping_method="generate"
    )
    return agent,memory