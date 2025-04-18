import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool

from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

@tool
def echo_tool(query: str) -> str:
    """Echo balik isi pertanyaan (dummy tool)"""
    return f"[Simulasi tool] {query}"
tools = [echo_tool]
load_dotenv()

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
        
        openai_api_base="https://api.together.xyz/v1",
        openai_api_key="12180e976ee41b0a7777732964c45a6768a4d7678c45417df37e6ba042a1bf48",
        temperature=0.7,
    )

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
        agent=AgentType.OPENAI_FUNCTIONS, # ⚠️ Bukan "MULTI_FUNCTIONS", karena Together belum full support
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent