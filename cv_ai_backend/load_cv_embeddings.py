
import sys
import json
import chromadb
from openai import OpenAI

sys.path.insert(0, ".")
from app.core.config import get_settings

def load_cv_embeddings():
    """Load your CV data into ChromaDB"""
    settings = get_settings()

    print("Initializing CV embedding loader...")

    # Initialize clients
    openai_client = OpenAI(api_key=settings.openai_api_key)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    # Create or get collection
    try:
        collection = chroma_client.get_collection("cv_chunks")
        if collection.count() > 0:
            print(f"Collection already has {collection.count()} documents")
            return
    except:
        collection = chroma_client.create_collection("cv_chunks")
        print("Created new ChromaDB collection")

    # Your CV data (replace with actual content from your CV)
    cv_sections = [
        """Aldo Hernandez Villanueva - Mexican economist with technical expertise in data analysis, 
        Python programming, SQL databases, and AI automation. Charismatic professional with 
        structured thinking and business acumen.""",

        """Technical Skills: Python programming, SQL databases, Tableau visualization, 
        Power BI dashboards, LangChain framework, OpenAI APIs, FastAPI development, 
        JavaScript, React, Pandas data analysis, Excel automation, statistical analysis.""",

        """Professional Experience: 2+ years in social media analytics and data processing. 
        Specialized in KPI modeling, sentiment analysis, cross-functional collaboration, 
        and stakeholder engagement. Process improvement specialist with automation expertise.""",

        """Education: Bachelor of Arts in Economics from Tecnologico de Monterrey. 
        Continuous learning in artificial intelligence, machine learning, and data science. 
        Strong foundation in economic theory and quantitative analysis.""",

        """Notable Projects: Social listening platform development using modern web technologies. 
        KPI automation and dashboard creation for business intelligence. Cross-campus analytics 
        implementation. Data-driven process optimization initiatives.""",

        """Communication Style: Uses expressions like 'la neta', 'te soy sincero', 'esto es clave'. 
        Direct, optimistic, and authentic communication. Structured thinking with creative and 
        technical problem-solving approach. Responsible for work quality and outcomes."""
    ]

    # Generate embeddings and store
    documents = []
    embeddings = []
    ids = []

    for i, section in enumerate(cv_sections):
        print(f"Processing section {i+1}/{len(cv_sections)}")

        try:
            response = openai_client.embeddings.create(
                model=settings.embedding_model,
                input=section.strip(),
                dimensions=settings.embedding_dimensions
            )

            documents.append(section.strip())
            embeddings.append(response.data[0].embedding)
            ids.append(f"cv_section_{i}")

        except Exception as e:
            print(f"Error processing section {i}: {e}")
            continue

    # Add to ChromaDB
    if documents:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids
        )

        print(f"Successfully loaded {len(documents)} CV sections!")
        print(f"Collection now has {collection.count()} documents")
    else:
        print("No documents were processed successfully")

if __name__ == "__main__":
    load_cv_embeddings()
