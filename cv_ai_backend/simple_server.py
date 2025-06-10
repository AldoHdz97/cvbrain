"""Simple server that works without CV embeddings"""

import sys
sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="CV-AI Backend",
    description="Simple test server",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {
        "message": "🧠 CV-AI Backend is running!",
        "status": "operational",
        "version": "3.0.0"
    }

@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "message": "Server is running",
        "embeddings_loaded": False
    }

@app.post("/api/v1/query") 
def test_query(data: dict):
    question = data.get("question", "")
    return {
        "answer": f"Test response for: {question}. (Full AI system will be available once embeddings are loaded)",
        "status": "test_mode",
        "processing_time_seconds": 0.1
    }

if __name__ == "__main__":
    print("🚀 Starting Simple CV-AI Backend...")
    print("🌐 API: http://127.0.0.1:8000")
    print("📖 Docs: http://127.0.0.1:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)