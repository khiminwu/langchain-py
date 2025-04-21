import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_google_vertexai import VertexAI
from google.cloud import aiplatform
from google.oauth2 import service_account
from langchain.schema import SystemMessage

load_dotenv()


def llm():
    
    # open router
    # return ChatOpenAI(
    #     model="google/gemini-2.0-flash-exp:free",  # model dari OpenRouter
    #     openai_api_base=os.getenv("OPENAI_API_BASE"),
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    #     temperature=0.7,
    # )

    # together
    return ChatOpenAI(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        # model="meta-llama/Llama-4-Scout-17B-16E-Instruct", # model dari Together ai
        max_tokens=1024,
        openai_api_base="https://api.together.xyz/v1",
        openai_api_key="12180e976ee41b0a7777732964c45a6768a4d7678c45417df37e6ba042a1bf48",
        temperature=0.7,
    )

    # return VertexAI(
    #     model_name="gemma-3-12b-it",
    #     project="devops-440402",
    #     location="us-central1",
    #     temperature=0.7,
    #     credentials=credentials,
    #     verbose=True,
    #     max_output_tokens=1024  # ✅ This is the correct parameter
    # )
    

