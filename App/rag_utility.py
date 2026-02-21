from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

# -------------- Embeding LLM ----------------

embedding_llm = OpenAIEmbeddings(model="text-embedding-3-small")

# ----------- PDF Retriever and Vector Store

# To store and retrive by thread_id
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# retriever
def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None



def load_existing_retriever(thread_id: str) -> Optional[Any]:
    """
    Load an existing Chroma retriever from disk if it exists.
    This is useful when restarting the app to restore retrievers.
    """
    persist_path = os.path.join("chatbot_chroma", str(thread_id))
    
    if not os.path.exists(persist_path):
        return None
    
    try:
        # Load existing vector store
        vector_store = Chroma(
            embedding_function=embedding_llm,
            persist_directory=persist_path,
            collection_name=str(thread_id)
        )
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 0.7}
        )
        
        # Store in memory for quick access
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        
        return retriever
    except Exception as e:
        print(f"Error loading retriever for thread {thread_id}: {e}")
        return None


# Get the PDF, split and store in vector_store
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a CHROMA retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    
    try:
        # load the document
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No documents loaded form PDF")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800, 
            chunk_overlap = 150, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # Ensure persist dir exists
        persist_path = os.path.join("chatbot_chroma", str(thread_id))
        os.makedirs(persist_path, exist_ok=True)

        # Vector Store
        vector_store = Chroma(
            embedding_function=embedding_llm,
            persist_directory=persist_path,
            collection_name=str(thread_id)
        )
        vector_store.add_documents(chunks)
        

        # Retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k":5, "lambda_mult":0.7}
        )

        # Store by thread_id
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        # Return success
        return {
            "success": True,
            "filename": filename or "uploaded.pdf",
            "pages": len(docs),
            "chunks": len(chunks),
            "message": f"Successfully processed {len(docs)} pages into {len(chunks)} chunks"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to process PDF: {str(e)}"
        }
    
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# Get thread Metadata
def get_thread_metadata(thread_id: str) -> dict:
    """Get metadata for a specific thread's PDF."""
    return _THREAD_METADATA.get(str(thread_id), {})

def has_pdf(thread_id: str) -> bool:
    """Check if a thread has an uploaded PDF."""
    return str(thread_id) in _THREAD_RETRIEVERS


def clear_thread_pdf(thread_id: str):
    """Remove PDF data for a specific thread."""
    thread_id_str = str(thread_id)
    
    # Remove from memory
    if thread_id_str in _THREAD_RETRIEVERS:
        del _THREAD_RETRIEVERS[thread_id_str]
    if thread_id_str in _THREAD_METADATA:
        del _THREAD_METADATA[thread_id_str]
    
    """
    Optionally delete from disk
    Do only if user asks
    """
    persist_path = os.path.join("chatbot_chroma", thread_id_str)
    if os.path.exists(persist_path):
        import shutil
        try:
            shutil.rmtree(persist_path)
        except Exception as e:
            print(f"Error removing PDF data: {e}")