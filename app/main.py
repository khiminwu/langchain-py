from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.agent import create_agent
from app.models import QueryRequest
app = FastAPI()


@app.post("/ask")
async def ask_agent(request: QueryRequest):
    try:
        agent = create_agent(request.session_id)
        response = agent.invoke(request.prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}