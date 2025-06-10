"""Load CV embeddings into ChromaDB"""

import sys
import json
import chromadb
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

sys.path.insert(0, ".")
from app.core.config import get_settings

def load_cv_data():
    """Load CV data and create embeddings"""
    settings = get_settings()
    
    print("🔧 Loading CV data...")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=settings.openai_api_key)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection("cv_chunks")
        print(f"📋 Found existing collection with {collection.count()} documents")
        if collection.count() > 0:
            print("✅ Collection already has data!")
            return
    except:
        collection = chroma_client.create_collection("cv_chunks")
        print("📋 Created new collection")
    
    # Load CV data (you'll need to point this to your actual CV file)
    cv_data_path = "../data/cv.json"  # Update this path to your CV file
    
    try:
        with open(cv_data_path, "r", encoding="utf-8") as f:
            cv_dict = json.load(f)
        cv_text = json.dumps(cv_dict, indent=2)
        print("✅ CV data loaded")
    except FileNotFoundError:
        print(f"❌ CV file not found at {cv_data_path}")
        print("💡 Please provide your CV data or update the path")
        return
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    docs = [Document(page_content=cv_text, metadata={"source": "cv.json"})]
    chunks = text_splitter.split_documents(docs)
    print(f"📝 Split into {len(chunks)} chunks")
    
    # Generate embeddings and add to collection
    documents = []
    embeddings = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
        
        # Get embedding
        response = openai_client.embeddings.create(
            model=settings.embedding_model,
            input=chunk.page_content,
            dimensions=settings.embedding_dimensions
        )
        
        documents.append(chunk.page_content)
        embeddings.append(response.data[0].embedding)
        ids.append(f"chunk_{i}")
    
    # Add to collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    
    print(f"✅ Added {len(chunks)} chunks to ChromaDB")
    print(f"📋 Collection now has {collection.count()} documents")

if __name__ == "__main__":
    load_cv_data()