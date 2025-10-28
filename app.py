from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kisan_agent import ProjectKisanAgent
import asyncio

app = FastAPI(title="Kisan Agent")


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

agent = ProjectKisanAgent()

@app.get("/")
def read_root():
    return {"message": "Kisan Agent is running!", "status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # this endpoint will be invoked when a user is chatting with SEC Agent

        response = await asyncio.to_thread(agent.chat, request.message)

        return ChatResponse(response=response["content"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}