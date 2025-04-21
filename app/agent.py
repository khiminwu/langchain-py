import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent,Tool, AgentType
# from langchain_core.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import VertexAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.google_serper.tool import GoogleSerperResults
from google.cloud import aiplatform
from google.oauth2 import service_account
from langchain.schema import SystemMessage

from langchain_experimental.tools.python.tool import PythonREPLTool
import torch

from app.llm import llm

from langchain.chains import ConversationChain

load_dotenv()

# credentials = service_account.Credentials.from_service_account_file(
#     "credential.json"
# )

# google_credentials_path = os.getenv("GOOGLE_CREDENTIAL")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
# Now your app should authenticate using the service account key.
# aiplatform.init(project="devops-440402", location="us-central1")

print(llm)

def is_greeting(query: str) -> bool:
    greetings = ["hi", "hello", "hey", "hai", "hola", "yo"]
    return query.strip().lower() in greetings

real_search_tool = GoogleSerperResults()

def safe_search(query: str) -> str:
    if is_greeting(query):
        return "Skipping search for greetings 😊"
    return real_search_tool.run(query)

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")



llm = llm()

# ✅ Dictionary untuk simpan memory per session
session_memories = {}


prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
        The following is a conversation between a helpful AI and a human:
        {chat_history}
        Human: {input}
        AI:"""
    )



# Create memory using Redis
def get_memory(session_id: str):
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    return ConversationBufferMemory(chat_memory=history, memory_key="chat_history", return_messages=False)


def chain_tool(session_id:str):
    memory = get_memory(session_id)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )
    return Tool(
        name="MemoryChatTool",
        func=chain.run,
        description="Useful for chatting with the AI and remembering past conversations.",
    )

def generate_positioning(input: str) -> str:
    prompt = f"""
Craft a brand positioning statement from the following info:

{input}

Use this format:
"For [target], [brand] is the [category] that [benefit], because [reason to believe]."
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def create_agent(session_id: str,llm_tools=None):
    
    memory = get_memory(session_id)
    
    if not memory.chat_memory.messages:
        memory.chat_memory.add_message(SystemMessage(
            content="I am a AI Senior Brand Strategy Manager, specialize in brand positioning, market analysis, storytelling, and communication strategy. Respond with strategic insights, market context, and persuasive branding advice."
        ))
   

    if llm_tools is None:
        llm_tools =[]
    
    
    tools = [chain_tool(session_id)]+llm_tools

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        # agent=AgentType.OPENAI_FUNCTIONS, # ⚠️ Bukan "MULTI_FUNCTIONS", karena Together belum full support
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=2,
        handle_parsing_errors=True,
        early_stopping_method="generate"
    )
    return agent,memory

def chat(prompt:str, session_id:str):
    tools = [
        Tool(
            name="SerpGoogleer Search",
            func=safe_search,
            description="Useful for answering questions about current events or recent info on the web",
            return_messages=True
        ),
    ]
    
    agent,memory = create_agent(session_id,tools)
    result = agent.invoke({"input": prompt})
    return {"response": result.get("output", "No output.")}
    # result = agent.run(prompt)
    # return result

# Tool example (positioning)
def strategy(input: str,session_id:str):
    
    tools = [
        Tool(
            name="Brand Positioning Generator",
            func=generate_positioning,
            description="Generate a brand positioning statement"
        )
    ]

    agent,memory = create_agent(session_id,tools)
    result = agent.invoke({"input": input})
    return {"response": result.get("output", "No output.")}



def analyze(prompt:str, session_id:str):
    tools = [
        # Tool(name="Brand Positioning Generator", func=generate_positioning, description="Generate brand positioning statement."),
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
    agent,memory = create_agent(session_id,tools)
    # print("summary_prompt",prompt)
    # result = agent.run(prompt)
    # return result
    result = agent.invoke({"input":prompt})
    return result.get("output", "No output generated.")
    # return prompt