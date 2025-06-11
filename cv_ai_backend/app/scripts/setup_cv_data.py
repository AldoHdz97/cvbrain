"""
CV Data Setup Script - Missing from cvbrain7L.md
Load cv.json into ChromaDB for the Ultimate CV Service
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_settings
from utils.chromadb_manager import UltimateChromaDBManager
from utils.connection_manager import UltimateConnectionManager

from openai import AsyncOpenAI

def chunk_cv_data(cv_data, chunk_size=600):
    """Very basic chunking: Split sections (summary, experience, projects, etc) into text chunks for embedding."""
    chunks = []

    def add_chunk(section, content):
        if content:
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = f"{section}: " + " | ".join([f"{k}: {v}" for k, v in item.items()])
                        chunks.append({"section": section, "text": text})
                    else:
                        chunks.append({"section": section, "text": f"{section}: {item}"})
            elif isinstance(content, dict):
                for k, v in content.items():
                    chunks.append({"section": f"{section}.{k}", "text": f"{section}.{k}: {v}"})
            else:
                chunks.append({"section": section, "text": f"{section}: {content}"})

    add_chunk("summary", cv_data.get("summary"))
    add_chunk("coreCompetences", cv_data.get("coreCompetences"))
    add_chunk("professionalExperience", cv_data.get("professionalExperience"))
    add_chunk("projects", cv_data.get("projects"))
    add_chunk("education", cv_data.get("education"))
    add_chunk("certifications", cv_data.get("certifications"))
    add_chunk("contact", cv_data.get("contact"))

    # Split long text fields if necessary (simple way)
    new_chunks = []
    for chunk in chunks:
        text = chunk["text"]
        while len(text) > chunk_size:
            split_at = text.rfind(" ", 0, chunk_size)
            if split_at == -1:
                split_at = chunk_size
            new_chunks.append({"section": chunk["section"], "text": text[:split_at]})
            text = text[split_at:]
        new_chunks.append({"section": chunk["section"], "text": text})
    return new_chunks

async def embed_and_upload(chunks, settings, chromadb_manager):
    """Embeds each chunk and uploads to ChromaDB."""
    # Initialize OpenAI async client
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    docs = []
    embeddings = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        # Get embedding
        resp = await openai_client.embeddings.create(
            model=settings.embedding_model,
            input=chunk["text"],
            dimensions=settings.embedding_dimensions,
            encoding_format="float"
        )
        embedding = resp.data[0].embedding
        docs.append(chunk["text"])
        embeddings.append(embedding)
        metadatas.append({"section": chunk["section"]})
        ids.append(f"cv_{chunk['section']}_{idx}")

    # Add all docs to ChromaDB
    await chromadb_manager.add_documents(
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ Uploaded {len(docs)} chunks to ChromaDB collection '{settings.chroma_collection_name}'")

async def load_cv_data():
    """Load CV data from JSON into ChromaDB"""
    print("🚀 Setting up CV data for Ultimate CV Service...")

    # Load CV JSON
    cv_path = Path(__file__).parent.parent / "data" / "cv.json"
    if not cv_path.exists():
        print(f"❌ CV file not found at {cv_path}")
        print("📝 Please add your cv.json file to app/data/cv.json")
        return False
    with open(cv_path, 'r', encoding='utf-8') as f:
        cv_data = json.load(f)
    print(f"✅ Loaded CV data for {cv_data['name']}")

    # Chunk CV
    print("🧩 Chunking CV sections...")
    chunks = chunk_cv_data(cv_data)
    print(f"🔹 Prepared {len(chunks)} text chunks for embedding.")

    # Initialize config and chromadb manager
    settings = get_settings()
    chromadb_manager = UltimateChromaDBManager(settings)
    await chromadb_manager.initialize()

    # Embed and upload chunks
    print("🔗 Embedding and uploading chunks...")
    await embed_and_upload(chunks, settings, chromadb_manager)
    print("🎉 CV setup complete.")

if __name__ == "__main__":
    asyncio.run(load_cv_data())
