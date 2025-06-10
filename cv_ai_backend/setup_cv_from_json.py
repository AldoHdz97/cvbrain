#!/usr/bin/env python3
"""
Fixed CV JSON Setup Script - ChromaDB Metadata Issue Resolved
Loads existing cv.json and populates ChromaDB with embeddings
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union
import hashlib

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import chromadb
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean metadata to ensure ChromaDB compatibility
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Cleaned metadata with only valid types
    """
    cleaned = {}
    
    for key, value in metadata.items():
        if value is None:
            # Skip None values
            continue
        elif isinstance(value, (str, int, float, bool)):
            # Valid ChromaDB types
            cleaned[key] = value
        elif isinstance(value, list):
            # Convert list to comma-separated string
            cleaned[key] = ", ".join(str(v) for v in value if v is not None)
        elif isinstance(value, dict):
            # Skip complex objects
            continue
        else:
            # Convert other types to string
            cleaned[key] = str(value)
    
    return cleaned


def find_cv_json() -> Path:
    """Find cv.json file in project directories"""
    
    possible_locations = [
        Path("data/cv.json"),
        Path("../data/cv.json"),
        Path("cv.json"),
        Path("../cv.json"),
        Path("app/data/cv.json"),
        Path("../embeddings/cv.json")
    ]
    
    for location in possible_locations:
        if location.exists():
            logger.info(f"✅ Found cv.json at: {location}")
            return location
    
    # If not found, ask user for location
    logger.error("❌ cv.json not found in common locations")
    logger.info("📁 Searched in:")
    for loc in possible_locations:
        logger.info(f"   - {loc.absolute()}")
    
    raise FileNotFoundError("cv.json not found. Please ensure cv.json exists in the project directory.")


def parse_cv_json(cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse cv.json and convert to text chunks suitable for embedding
    
    Args:
        cv_data: Parsed JSON data from cv.json
        
    Returns:
        List of text chunks with metadata
    """
    
    chunks = []
    
    def add_chunk(text: str, section: str, subsection: str = None, metadata: Dict = None):
        """Helper to add a chunk with consistent metadata"""
        if not text or not text.strip():
            return
        
        chunk_metadata = {
            'section': section,
            'chunk_index': len(chunks),
            'char_count': len(text),
            'source': 'cv.json'
        }
        
        if subsection:
            chunk_metadata['subsection'] = subsection
        
        if metadata:
            # Only add non-None values
            for k, v in metadata.items():
                if v is not None:
                    chunk_metadata[k] = v
        
        # Clean metadata for ChromaDB
        chunk_metadata = clean_metadata(chunk_metadata)
        
        chunks.append({
            'text': text.strip(),
            'metadata': chunk_metadata
        })
    
    # Process different sections of the CV
    
    # 1. Personal Information & Summary
    if 'name' in cv_data:
        name = cv_data['name']
        summary_text = f"I am {name}"
        
        if 'summary' in cv_data:
            summary_text += f".\n\n{cv_data['summary']}"
        
        add_chunk(summary_text, "Personal Information", "Summary")
    
    # 2. Contact Information
    if 'contact' in cv_data:
        contact_data = cv_data['contact']
        contact_info = []
        
        for field in ['location', 'phone', 'email', 'linkedin', 'tableauPublic']:
            if field in contact_data and contact_data[field]:
                contact_info.append(f"{field.title()}: {contact_data[field]}")
        
        if contact_info:
            add_chunk("\n".join(contact_info), "Contact Information")
    
    # 3. Core Competences (Skills)
    if 'coreCompetences' in cv_data:
        competences = cv_data['coreCompetences']
        
        for category, skills in competences.items():
            if isinstance(skills, list) and skills:
                skills_text = f"{category.replace('Skills', ' Skills').title()}:\n"
                skills_text += "\n".join(f"- {skill}" for skill in skills)
                add_chunk(skills_text, "Skills", category.title())
    
    # 4. Professional Experience
    if 'professionalExperience' in cv_data:
        experiences = cv_data['professionalExperience']
        
        for i, exp in enumerate(experiences):
            exp_text = []
            
            if 'role' in exp:
                exp_text.append(f"Position: {exp['role']}")
            
            if 'company' in exp:
                exp_text.append(f"Company: {exp['company']}")
            
            if 'startDate' in exp or 'endDate' in exp:
                start = exp.get('startDate', 'N/A')
                end = exp.get('endDate', 'Present')
                exp_text.append(f"Duration: {start} - {end}")
            
            if 'bullets' in exp and isinstance(exp['bullets'], list):
                exp_text.append("Key Responsibilities and Achievements:")
                for bullet in exp['bullets']:
                    exp_text.append(f"• {bullet}")
            
            if exp_text:
                full_text = "\n".join(exp_text)
                company_name = exp.get('company', f'Experience {i+1}')
                add_chunk(full_text, "Experience", company_name, {
                    'experience_index': i,
                    'company': exp.get('company'),
                    'role': exp.get('role')
                })
    
    # 5. Projects
    if 'projects' in cv_data:
        projects = cv_data['projects']
        
        for i, project in enumerate(projects):
            proj_text = []
            
            if 'title' in project:
                proj_text.append(f"Project: {project['title']}")
            
            if 'technology' in project:
                proj_text.append(f"Technologies: {project['technology']}")
            
            if 'description' in project:
                proj_text.append(f"Description: {project['description']}")
            
            if 'bullets' in project and isinstance(project['bullets'], list):
                proj_text.append("Key Features and Achievements:")
                for bullet in project['bullets']:
                    proj_text.append(f"• {bullet}")
            
            if proj_text:
                full_text = "\n".join(proj_text)
                project_name = project.get('title', f'Project {i+1}')
                add_chunk(full_text, "Projects", project_name, {
                    'project_index': i,
                    'project_name': project.get('title'),
                    'technology': project.get('technology')
                })
    
    # 6. Education
    if 'education' in cv_data:
        education_data = cv_data['education']
        
        for i, edu in enumerate(education_data):
            edu_text = []
            
            if 'degree' in edu:
                edu_text.append(f"Degree: {edu['degree']}")
            
            if 'institution' in edu:
                edu_text.append(f"Institution: {edu['institution']}")
            
            if 'startYear' in edu and 'endYear' in edu:
                edu_text.append(f"Years: {edu['startYear']} - {edu['endYear']}")
            
            if 'notes' in edu:
                edu_text.append(f"Notes: {edu['notes']}")
            
            if edu_text:
                full_text = "\n".join(edu_text)
                institution_name = edu.get('institution', f'Education {i+1}')
                add_chunk(full_text, "Education", institution_name, {
                    'education_index': i,
                    'institution': edu.get('institution'),
                    'degree': edu.get('degree')
                })
    
    # 7. Certifications
    if 'certifications' in cv_data:
        certs = cv_data['certifications']
        
        if isinstance(certs, list) and certs:
            cert_text = "Certifications:\n" + "\n".join(f"• {cert}" for cert in certs)
            add_chunk(cert_text, "Certifications")
    
    # 8. Handle any other top-level fields
    exclude_fields = {
        'name', 'contact', 'summary', 'coreCompetences', 'professionalExperience', 
        'projects', 'education', 'certifications'
    }
    
    for key, value in cv_data.items():
        if key not in exclude_fields and value:
            if isinstance(value, str):
                add_chunk(f"{key.title()}: {value}", "Additional Information", key)
            elif isinstance(value, list) and value:
                list_text = f"{key.title()}:\n" + "\n".join(f"• {item}" for item in value)
                add_chunk(list_text, "Additional Information", key)
    
    logger.info(f"✅ Parsed CV JSON into {len(chunks)} chunks")
    
    # Log section distribution
    section_counts = {}
    for chunk in chunks:
        section = chunk['metadata']['section']
        section_counts[section] = section_counts.get(section, 0) + 1
    
    logger.info("📊 Section distribution:")
    for section, count in section_counts.items():
        logger.info(f"   - {section}: {count} chunks")
    
    return chunks


async def populate_chromadb_from_json():
    """Populate ChromaDB with embeddings from existing cv.json"""
    
    logger.info("🚀 Starting CV JSON setup and ChromaDB population...")
    
    # 1. Find and load cv.json
    try:
        cv_json_path = find_cv_json()
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    logger.info(f"📖 Loading CV data from {cv_json_path}...")
    
    try:
        with open(cv_json_path, 'r', encoding='utf-8') as f:
            cv_data = json.load(f)
        logger.info("✅ CV JSON loaded successfully")
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format in {cv_json_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to read {cv_json_path}: {e}")
        return False
    
    # 2. Parse CV data into chunks
    logger.info("🔪 Parsing CV JSON into chunks...")
    try:
        chunks = parse_cv_json(cv_data)
        if not chunks:
            logger.error("❌ No chunks generated from CV data")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to parse CV JSON: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Initialize OpenAI client
    try:
        from app.core.config import get_settings
        settings = get_settings()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.info("✅ OpenAI client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        logger.error("💡 Make sure your OpenAI API key is set in the .env file")
        return False
    
    # 4. Initialize ChromaDB
    logger.info("🗄️  Initializing ChromaDB...")
    embeddings_dir = Path("../embeddings/chroma")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=str(embeddings_dir))
    
    # Create collection name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_name = f"cv_chunks_{timestamp}"
    
    # Delete existing collection if it exists
    try:
        existing_collections = chroma_client.list_collections()
        for collection in existing_collections:
            logger.info(f"🗑️  Deleting existing collection: {collection.name}")
            chroma_client.delete_collection(collection.name)
    except:
        pass
    
    # Create new collection
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "description": "CV chunks from cv.json with embeddings",
            "source_file": str(cv_json_path),
            "created_at": timestamp
        }
    )
    
    logger.info(f"✅ Created ChromaDB collection: {collection_name}")
    
    # 5. Generate embeddings and populate ChromaDB
    logger.info("🔮 Generating embeddings and populating ChromaDB...")
    
    batch_size = 5  # Smaller batch size for better error handling
    total_chunks = len(chunks)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [chunk['text'] for chunk in batch]
        batch_metadatas = [chunk['metadata'] for chunk in batch]
        batch_ids = [f"chunk_{j}" for j in range(i, min(i + batch_size, total_chunks))]
        
        try:
            # Debug: Log metadata for first batch
            if i == 0:
                logger.info("🔍 Sample metadata structure:")
                for j, metadata in enumerate(batch_metadatas):
                    logger.info(f"   Chunk {j}: {metadata}")
            
            # Generate embeddings
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # Add to ChromaDB with cleaned metadata
            collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            logger.info(f"✅ Processed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
            logger.error(f"Problematic metadata: {batch_metadatas}")
            return False
    
    # 6. Update configuration with new collection name
    logger.info("⚙️  Updating configuration...")
    
    env_file_path = Path(".env")
    if env_file_path.exists():
        # Read current .env
        with open(env_file_path, "r") as f:
            env_content = f.read()
        
        # Update collection name
        if "CV_AI_CHROMA_COLLECTION_NAME=" in env_content:
            # Replace existing
            lines = env_content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("CV_AI_CHROMA_COLLECTION_NAME="):
                    lines[i] = f"CV_AI_CHROMA_COLLECTION_NAME={collection_name}"
                    break
            env_content = '\n'.join(lines)
        else:
            # Add new line
            env_content += f"\nCV_AI_CHROMA_COLLECTION_NAME={collection_name}\n"
        
        # Write back
        with open(env_file_path, "w") as f:
            f.write(env_content)
        
        logger.info(f"✅ Updated .env with collection name: {collection_name}")
    
    # 7. Test the setup
    logger.info("🧪 Testing ChromaDB setup...")
    
    test_queries = [
        "What are your technical skills?",
        "Tell me about your work experience", 
        "What projects have you worked on?"
    ]
    
    for query in test_queries:
        try:
            test_results = collection.query(
                query_texts=[query],
                n_results=2
            )
            
            if test_results['documents'] and test_results['documents'][0]:
                logger.info(f"✅ Test query '{query[:30]}...' returned {len(test_results['documents'][0])} results")
            else:
                logger.warning(f"⚠️  Test query '{query[:30]}...' returned no results")
        except Exception as e:
            logger.error(f"❌ Test query failed: {e}")
            return False
    
    logger.info(f"📊 Final ChromaDB stats:")
    logger.info(f"   - Collection: {collection_name}")
    logger.info(f"   - Total documents: {collection.count()}")
    logger.info(f"   - Source file: {cv_json_path}")
    
    return True


def main():
    """Main function to set up CV data from JSON and populate ChromaDB"""
    
    logger.info("🎯 CV-AI JSON Data Setup Script (FIXED)")
    logger.info("=" * 50)
    
    try:
        success = asyncio.run(populate_chromadb_from_json())
        
        if success:
            logger.info("=" * 50)
            logger.info("🎉 Setup completed successfully!")
            logger.info("")
            logger.info("📋 Next steps:")
            logger.info("1. python run.py")
            logger.info("2. Visit http://localhost:8000/docs")
            logger.info("3. Try queries about your CV!")
            logger.info("")
            logger.info("🔍 Health check: http://localhost:8000/api/v1/health")
            logger.info("🧪 Example queries:")
            logger.info("   - 'What are your main technical skills?'")
            logger.info("   - 'Tell me about your work at Swarm Data'")
            logger.info("   - 'What projects have you built?'")
        else:
            logger.error("❌ Setup failed. Please check the errors above.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("👋 Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())