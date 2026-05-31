import os
import uuid
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import Chunk
from src.utils.logger import logger

class DocumentProcessorService:
    """Parses source files using LangChain Document Loaders and splits text using Text Splitters."""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self, notebook_id: str, document_id: str, file_path: str) -> List[Chunk]:
        """Loads a document using LangChain loaders and splits it into semantic chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        _, ext = os.path.splitext(file_path.lower())
        
        # 1. Initialize LangChain Document Loader based on file type
        logger.info(f"Using LangChain Document Loader for: {file_path}")
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}. We support .pdf, .txt, and .md.")

        # Load raw documents
        raw_docs = loader.load()
        logger.info(f"Successfully loaded document with {len(raw_docs)} original raw parts/pages.")

        # 2. Split document using LangChain Text Splitter
        logger.info(f"Using LangChain RecursiveCharacterTextSplitter (size={self.chunk_size}, overlap={self.chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        split_docs = splitter.split_documents(raw_docs)
        logger.info(f"Document split into {len(split_docs)} semantic chunks.")

        # 3. Map LangChain Document objects to our Domain Chunk models
        chunks = []
        for idx, doc in enumerate(split_docs):
            # Fetch page number metadata (LangChain's page is 0-indexed for PDFs, convert to 1-indexed)
            # If not present (e.g. TXT/MD files), defaults to page 0
            page_idx = doc.metadata.get("page", -1)
            page_number = page_idx + 1 if page_idx >= 0 else 0
            
            chunk_id = f"chk_{str(uuid.uuid4())}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    notebook_id=notebook_id,
                    text=doc.page_content,
                    page_number=page_number,
                    chunk_index=idx
                )
            )
            
        return chunks
