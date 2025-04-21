from fastapi import FastAPI, Request,File, UploadFile,Form
from pydantic import BaseModel
from app.agent import create_agent, chat,analyze,strategy
from app.models import QueryRequest
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os, shutil
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(session_id: str = Form(...),
                file: Optional[UploadFile] = File(None),
                prompt: str = Form(""),
                brand_name: str = Form(""),
                target_audience: str = Form(""),
                category: str = Form(""),
                benefit: str = Form(""),
                reason: str = Form(""),
                type: str = Form("chat")):
    try:
        
        if(type=="analyze"):
            if file is None:
                return {"error": "File is required for analysis."}
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
            summary_prompt = f"Please summarize the following document:\n\n{full_text[:4000]}"
            # print("summary_prompt",summary_prompt)
            result = analyze(summary_prompt,session_id)
            return {"response": result}
            # agent,memory = create_agent(session_id)
            # result = agent.invoke({"input":prompt+"\n\n"+full_text[:4000]})

            # os.remove(file_path)
            # return {"response": type}
        elif(type=="strategy"):
            prompt = (
                f"Brand: {brand_name}\n"
                f"Target Audience: {target_audience}\n"
                f"Category: {category}\n"
                f"Benefit: {benefit}\n"
                f"Reason to Believe: {reason}"
            )
            result = strategy(prompt,session_id)
            return {"response": result}
        else:
            result = chat(prompt,session_id)
            return {"response": result}

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


@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok", "message": "Service is healthy"}, status_code=200)