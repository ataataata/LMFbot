# backend.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ollama import Client
import numpy as np
import json
import os
import traceback

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

MODEL = "qwen2.5vl:3b"
K = 3

app = FastAPI()
oll = Client()

try:
    if os.path.exists("qa.json"):
        qa = json.load(open("qa.json"))
        print(f"Loaded {len(qa)} Q-A pairs")
    else:
        print("Warning: qa.json not found, using empty list")
        qa = []
    
    if os.path.exists("vecs.npy"):
        vecs = np.load("vecs.npy")
        print(f"Loaded vectors with shape: {vecs.shape}")
    else:
        print("Warning: vecs.npy not found, using empty array")
        vecs = np.array([])
        
except Exception as e:
    print(f"Error loading data: {e}")
    qa = []
    vecs = np.array([])

def embed(text: str):
    try:
        e = oll.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
        return np.asarray(e, dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        return np.array([])

def retrieve(query: str, k=K):
    try:
        if len(qa) == 0 or len(vecs) == 0:
            print("No data available for retrieval")
            return []
            
        qv = embed(query)
        if len(qv) == 0:
            return []
            
        sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv) + 1e-9)
        top = sims.argsort()[-k:][::-1]
        return [qa[i] for i in top]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

SYS_MSG = "You are the website chatbot. Use the shown Q-A pairs when possible."

def build_messages(user_msg: str):
    try:
        ctx = retrieve(user_msg)
        if ctx:
            kb = "\n".join(f"Q: {c['q']}\nA: {c['a']}" for c in ctx)
            system_content = SYS_MSG + "\n\n" + kb
        else:
            system_content = SYS_MSG + "\n\nNo relevant context found."
            
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_msg},
        ]
    except Exception as e:
        print(f"Message building error: {e}")
        return [
            {"role": "system", "content": SYS_MSG},
            {"role": "user", "content": user_msg},
        ]

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        print(f"Received message: {req.message}")
        
        # Build messages
        messages = build_messages(req.message)
        print(f"Built messages: {len(messages)} messages")
        
        try:
            res = oll.chat(model=MODEL, messages=messages, stream=False)
            print(f"Ollama response received")
        except Exception as ollama_error:
            print(f"Ollama error: {ollama_error}")
            return ChatResponse(answer=f"I'm having trouble connecting to the AI model. Error: {str(ollama_error)}")
        
        if res and "message" in res and "content" in res["message"]:
            raw_answer = res["message"]["content"]
            print(f"Raw answer: {raw_answer[:200]}...")
            
     
            if "</think>" in raw_answer:
                answer = raw_answer.split("</think>")[-1].strip()
                print(f"Extracted answer after </think>: {answer[:100]}...")
            else:
                answer = raw_answer.strip()
                
            if not answer or answer.isspace():
                answer = "I understand your message, but I'm having trouble formulating a response."
                
        else:
            print(f"Unexpected response format: {res}")
            answer = "I received an unexpected response format from the AI model."
        
        return ChatResponse(answer=answer)
        
    except Exception as e:
        error_msg = f"Backend error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return ChatResponse(answer=error_msg)

@app.get("/health")
async def health_check():
    try:
        models = oll.list()
        return {"status": "healthy", "models_available": len(models.get("models", []))}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)