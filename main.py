import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Query
from mistralai import Mistral
import fitz
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import logging

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
embedding_model = "mistral-embed"
client = Mistral(api_key=api_key)

app = FastAPI(
    title="PDF RAG Pipeline API",
    description="API to upload PDF files, ingest their content, and query them using a RAG pipeline with Mistral AI",
    version="0.1.0"
)

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    GREETING = "greeting"
    KNOWLEDGE_QUERY = "knowledge_query"
    CHITCHAT = "chitchat"
    SYSTEM_COMMAND = "system_command"
    OTHER = "other"

def classify_query_intent(query: str) -> Tuple[QueryIntent, str]:
    """Classifies the intent of a user query using LLM"""
    classification_prompt = f"""Classify the following user query into one of these categories:

1. GREETING - Simple greetings, pleasantries, or social interactions (hi, hello, how are you, goodbye, thanks)
2. KNOWLEDGE_QUERY - Questions seeking specific information that would benefit from document retrieval
3. CHITCHAT - Casual conversation, opinions, or general questions not requiring document search
4. SYSTEM_COMMAND - Questions about the system itself, its capabilities, or status

Query: "{query}"

Respond with ONLY the category name and a brief reason (max 10 words).
Format: CATEGORY|reason"""

    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are a query intent classifier. Be concise and accurate."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        parts = result.split("|")
        
        if len(parts) >= 1:
            intent_str = parts[0].strip().upper()
            reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
            
            # Map to enum
            intent_map = {
                "GREETING": QueryIntent.GREETING,
                "KNOWLEDGE_QUERY": QueryIntent.KNOWLEDGE_QUERY,
                "CHITCHAT": QueryIntent.CHITCHAT,
                "SYSTEM_COMMAND": QueryIntent.SYSTEM_COMMAND
            }
            
            intent = intent_map.get(intent_str, QueryIntent.KNOWLEDGE_QUERY)
            logger.info(f"Query '{query}' classified as {intent.value}: {reason}")
            return intent, reason
            
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Default to knowledge query to ensure RAG is used when uncertain
        return QueryIntent.KNOWLEDGE_QUERY, "Classification failed"

def handle_conversational_query(query: str, intent: QueryIntent) -> str:
    """Handles non-RAG queries directly with the LLM"""
    system_messages = {
        QueryIntent.GREETING: "You are a friendly assistant. Respond warmly and briefly to greetings.",
        QueryIntent.CHITCHAT: "You are a helpful conversational assistant. Keep responses concise and friendly.",
        QueryIntent.SYSTEM_COMMAND: "You are a system assistant. Explain the PDF RAG system's capabilities when asked."
    }
    
    system_content = system_messages.get(intent, "You are a helpful assistant.")
    
    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in conversational response: {e}")
        return "I'm sorry, I encountered an error processing your request."
    
document_store: Dict[str, Any] = {
    "chunks": [],  # List of {"text": "chunk_text", "source": "filename_chunk_idx"}
    "embeddings": [],  # List of numpy arrays (embeddings)
    "matrix": None,  # Combined numpy matrix of all embeddings
}

def extract_text_from_pdf(file_content: bytes) -> str:
    """Extracts text from a PDF file using PyMuPDF"""
    text = ""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") or ""
        doc.close()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Chunks text into smaller pieces with overlap."""
    chunks = []
    start = 0
    # Ensure text is not empty or just whitespace
    if not text or not text.strip():
        return []
        
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - chunk_overlap # Move window with overlap
        # Ensure we don't go past the end if overlap is large or text is short
        if start >= len(text): 
            break
            
    # Filter out any chunks that might be empty or only whitespace after processing
    return [chunk for chunk in chunks if chunk.strip()]

def get_embeddings(texts: List[str], client_instance: Mistral, model: str) -> Optional[np.ndarray]:
    """Generates embeddings for a list of texts using Mistral AI."""
    if not texts:
        return None
    try:
        # Corrected: Use client.embeddings method
        embeddings_response = client_instance.embeddings(
            model=model,
            input=texts # 'input' for list of strings
        )
        # Ensure the response structure is as expected
        if embeddings_response.data and all(hasattr(item, 'embedding') for item in embeddings_response.data):
            return np.array([item.embedding for item in embeddings_response.data], dtype=np.float32)
        else:
            print(f"Unexpected embeddings response structure: {embeddings_response}")
            return None
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        return None

def rebuild_matrix() -> None:
    if document_store["embeddings"]:
        document_store["matrix"] = np.vstack(document_store["embeddings"]).astype(np.float32)

def embed_text(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=embedding_model, inputs=texts)
    return np.array([r.embedding for r in resp.data], dtype=np.float32)

def cosine_similarity_search(query_embedding: np.ndarray, doc_matrix: np.ndarray, top_k: int = 5) -> List[int]:
    """Performs cosine similarity search and returns indices of top_k results."""
    if query_embedding is None or doc_matrix is None or doc_matrix.size == 0:
        return []

    # Normalize query and document embeddings
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_matrix_norm = doc_matrix / np.linalg.norm(doc_matrix, axis=1, keepdims=True)
    
    # Calculate cosine similarities
    similarities = np.dot(doc_matrix_norm, query_norm.T)
    
    # Get top_k indices
    # Using argsort for simplicity here. For very large matrices, argpartition could be faster.
    if top_k >= len(similarities):
        # If top_k is larger than or equal to available items, return all sorted
        sorted_indices = np.argsort(similarities)[::-1]
    else:
        # Get indices of the top_k largest similarities
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
    return sorted_indices.tolist()

def transform_query_for_search(original_query: str, client_instance: Mistral, model: str) -> str:
    """Transforms the user query for better search retrieval using an LLM."""
    system_prompt = (
        """You are a query optimization assistant. Your task is to rephrase the following user's question 
        into a concise and effective search query, suitable for a semantic search against a document database. 
        Focus on extracting key entities, concepts, and the core intent of the question. 
        Remove conversational fluff. Output only the refined search query."""
    )
    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": original_query}
            ],
            temperature=0.1 # Lower temperature for more deterministic and focused output
        )
        if response.choices and response.choices[0].message:
            transformed_query = response.choices[0].message.content.strip()
            logger.info(f"Original query: '{original_query}' | Transformed query: '{transformed_query}'")
            return transformed_query
        else:
            logger.warning(f"Query transformation failed to produce content for: '{original_query}'")
            return original_query # Fallback to original
    except Exception as e:
        logger.error(f"Error during query transformation for '{original_query}': {e}")
        return original_query # Fallback to original query in case of error

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    processed_files_summary = []
    new_chunks_for_embedding = [] # Tuples of (text, source_filename)

    for f in files:
        if f.content_type != "application/pdf":
            processed_files_summary.append({"filename": f.filename, "status": "error", "message": "Not a PDF"})
            continue
        
        try:
            contents = await f.read()
            extracted_text = extract_text_from_pdf(contents)

            if not extracted_text.strip():
                processed_files_summary.append({"filename": f.filename, "status": "warning", "message": "No text extracted or empty PDF."})
                continue

            text_chunks = chunk_text(extracted_text) 
            
            for i, chunk_content in enumerate(text_chunks):
                source_identifier = f"{f.filename}_chunk_{i+1}"
                new_chunks_for_embedding.append((chunk_content, source_identifier))
            
            processed_files_summary.append({"filename": f.filename, "status": "processed", "chunks_created": len(text_chunks)})

        except Exception as e:
            processed_files_summary.append({"filename": f.filename, "status": "error", "message": f"Failed to process: {str(e)}"})
        finally:
            await f.close()

    if not new_chunks_for_embedding:
        return {"message": "No new text to process from uploaded files.", "files_status": processed_files_summary}

    texts_to_embed = [item[0] for item in new_chunks_for_embedding]
    
    try:
        embeddings_response = client.embeddings.create(
            model=embedding_model,
            inputs=texts_to_embed
        )
        generated_embeddings = [np.array(item.embedding) for item in embeddings_response.data]
        
        # Append to our in-memory store
        for i, (text_content, source_id) in enumerate(new_chunks_for_embedding):
            document_store["chunks"].append({"text": text_content, "source": source_id})
            document_store["embeddings"].append(generated_embeddings[i])
            
    except Exception as e:
        return {"error": f"Failed to generate or store embeddings: {str(e)}", "files_status": processed_files_summary}

    return {
        "message": "Files processed and ingested successfully.",
        "processed_summary": processed_files_summary,
        "new_chunks_added": len(new_chunks_for_embedding),
        "total_chunks_in_store": len(document_store["chunks"]),
    }

@app.post("/query")
async def query_llm(
    prompt: str = Query(..., description="The user's query to the LLM"),
    top_k_results: int = Query(5, description="Number of top similar chunks to retrieve for context.")
    ):
    
    intent, intent_reason = classify_query_intent(prompt)
    
    if intent in [QueryIntent.GREETING, QueryIntent.CHITCHAT, QueryIntent.SYSTEM_COMMAND]:
        response = handle_conversational_query(prompt, intent)
        return {
            "response": response,
            "intent": intent.value,
            "intent_reason": intent_reason,
            "used_rag": False
        }
    
    if not document_store["chunks"]:
        return {
            "response": "I don't have any documents in my knowledge base yet. Please upload some PDFs first so I can help answer your questions!",
            "intent": intent.value,
            "intent_reason": intent_reason,
            "used_rag": False,
            "error": "No documents available"
        }
    
    try:
        search_query = transform_query_for_search(prompt, client, model)
        query_embedding_response = client.embeddings.create(
            model=embedding_model,
            inputs=[search_query]
        )
        query_embedding = np.array(query_embedding_response.data[0].embedding)

        if document_store["matrix"] is None:
            rebuild_matrix()


        top_k_indices = cosine_similarity_search(
            query_embedding,
            document_store["matrix"],
            top_k=top_k_results
        )

        retrieved_chunks = []
        for idx in top_k_indices:
            chunk_data = document_store["chunks"][idx]
            retrieved_chunks.append(f"[Source: {chunk_data['source']}]\n{chunk_data['text']}")

        context = "\n\n---\n\n".join(retrieved_chunks)

        rag_prompt = f"""Based on the following context from the knowledge base, please answer the user's question.

Context:
{context}

User Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing."""

        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context from PDF documents."
                },
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ]
        )
        return {
            "response": chat_response.choices[0].message.content,
            "intent": intent.value,
            "intent_reason": intent_reason,
            "used_rag": True,
            "sources_used": [document_store["chunks"][idx]["source"] for idx in top_k_indices],
            "query_transformed": search_query
        }
        
    except Exception as e:
        logger.error(f"Error during RAG query processing: {e}")
        return {
            "response": "I encountered an error while searching the knowledge base. Please try again.",
            "intent": intent.value,
            "intent_reason": intent_reason,
            "used_rag": True,
            "error": str(e)
        }