from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.agent import create_agent
from app.models import QueryRequest
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_agent(request: QueryRequest):
    try:
        agent = create_agent(request.session_id)
        response = agent.invoke(request.prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok", "message": "Service is healthy"}, status_code=200)

@app.get("/")
def health_check():
    return JSONResponse(content={"status": "ok", "message": "Service is healthy"}, status_code=200)