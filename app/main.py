from fastapi import FastAPI, Request,File, UploadFile,Form
from pydantic import BaseModel
from app.agent import create_agent
from app.models import QueryRequest
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os, shutil


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(request: QueryRequest):
    try:
        #request.prompt
        agent,memory = create_agent(request.session_id)
        response = agent.invoke({"input":request.prompt})
        return {"response": response.get("output", "No output generated.")}
    except Exception as e:
        return {"error": str(e)}

@app.post("/history")
async def history(request: QueryRequest):
    try:
        #request.prompt
        agent,memory = create_agent(request.session_id)
        history = [
            {"role": msg.type, "content": msg.content}
            for msg in memory.chat_memory.messages
        ]

        return {"response": history}
    except Exception as e:
        return {"error": str(e)}

# ==== Upload and summarize ====
@app.post("/upload")
async def upload_and_summarize(file: UploadFile = File(...),
                            session_id: str = Form(...),
                            prompt: str = Form("Please summarize the following document:")
                            ):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Determine loader
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()
    full_text = "\n".join([doc.page_content for doc in docs])

    # Summarize with prompt
    # summary_prompt = f"Please summarize the following document:\n\n{full_text[:4000]}"
    # result = ask_agent(summary_prompt)
    agent,memory = create_agent(session_id)
    result = agent.invoke({"input":prompt+"\n\n"+full_text[:4000]})

    os.remove(file_path)
    return {"response": result.get("output", "No output generated.")}

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok", "message": "Service is healthy"}, status_code=200)