"""
Vector Database Management for Legal RAG Pipeline
================================================

This module provides comprehensive vector database functionality for the Legal
RAG Pipeline, including document loading, text extraction, OCR processing,
and FAISS-based vector storage with GPU acceleration support.

Features:
- Multi-format document loading (PDF, web pages, text files)
- OCR fallback for scanned documents and images
- FAISS vector store with GPU acceleration
- Chunking and embedding of documents
- Web document scraping and processing
- Local file caching and management


"""

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from faiss import StandardGpuResources, index_cpu_to_gpu
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.generation.model_provider import get_embedding_model
from langchain_core.documents import Document
import os
import pytesseract
from pdf2image import convert_from_path
import webbrowser
from PIL import Image
import pytesseract
import requests
import time
from src.generation.model_provider import get_embedding_model
import numpy as np
from uuid import uuid4
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Ensure local directory exists for storing downloaded PDFs
local_file_dir = "app_data/local_pdfs"
os.makedirs(local_file_dir, exist_ok=True)


def extract_text_with_ocr(file_path):
    """
    Extract text from PDF or image files using OCR technology.
    
    This function uses Tesseract OCR to extract text from scanned documents
    and images when standard text extraction methods fail.
    
    Args:
        file_path (str): Path to the PDF or image file
        
    Returns:
        str: Extracted text content
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == ".pdf":
            # Convert PDF pages to images and apply OCR
            images = convert_from_path(file_path)
            for img in images:
                text += pytesseract.image_to_string(img)
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            # Apply OCR directly to image files
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
    except Exception as e:
        logger.info(f"OCR extraction failed for {file_path}: {e}")
    
    return text


def load_source_with_fallback(source):
    """
    Load document text with automatic fallback to OCR if needed.
    
    This function attempts to load documents using standard methods
    (PyPDFLoader for PDFs, web scraping for URLs) and falls back to
    OCR processing if standard extraction fails or produces poor results.
    
    Supports multiple document formats:
    - PDF files (local and web URLs)
    - Image files (PNG, JPG, JPEG, BMP, TIFF)
    - Text files (UTF-8 encoded)
    
    Args:
        source (str): Path to the document file or URL
        
    Returns:
        Document: LangChain Document object containing extracted text and metadata
    """
    doc = None
    local_file_path = os.path.join(local_file_dir, source.split("/")[-1])
    ext = os.path.splitext(source)[1].lower()
    previous_cache_from_web = f"{local_file_path}.txt"

    if os.path.exists(previous_cache_from_web):
        source = previous_cache_from_web
        ext = ".txt"

    if ext == ".pdf":
        # Handle web PDF downloads
        if source.startswith("https"):
            logger.info(f"\tLoading PDF from URL: {source}")
            # Download the PDF if it doesn't exist locally
            if not os.path.exists(local_file_path):
                headers = {
                            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:139.0) Gecko/20100101 Firefox/139.0",
                            }

                response = requests.get(source, headers=headers, stream=True)
                if response.status_code == 200:
                    os.makedirs(local_file_dir, exist_ok=True)
                    with open(local_file_path, "wb") as pdf_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            pdf_file.write(chunk)
                    logger.info(f"PDF downloaded successfully to {local_file_path}")
                    ocr_text = extract_text_with_ocr(local_file_path)
                    if ocr_text.strip():
                        doc = Document(page_content=ocr_text, metadata={'source': source, "ocr_applied": True})
    
                elif response.status_code == 404:
                    source = source.replace(".pdf", ".html")
                    web_doc = WebBaseLoader(source, continue_on_failure=True).load()[0]
                    web_doc.page_content = web_doc.page_content.strip()
                    with open(previous_cache_from_web, "w", encoding="utf-8") as f:
                        f.write(web_doc.page_content)
                    return web_doc

                elif response.status_code == 429:
                    time.sleep(10)  # Wait for the browser to close
                    return load_source_with_fallback(source)

            else:
                logger.info(f"PDF already exists locally at {local_file_path}")
                ocr_text = extract_text_with_ocr(local_file_path)
                if ocr_text.strip():
                    doc = Document(page_content=ocr_text, metadata={'source': source, "ocr_applied": True})
    
        else:
            logger.info(f"\tUsing PyPDFLoader for local PDF")
            loader = PyPDFLoader(source)
            doc = loader.load()[0]

        # If no text extracted, fallback to OCR
        if not doc or not doc.page_content.strip():
            logger.info(f"\tNo text extracted from PDF, applying OCR")
            ocr_text = extract_text_with_ocr(source)
            if ocr_text.strip():
                doc = Document(page_content=ocr_text, metadata={'source': source, "ocr_applied": True})
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        ocr_text = extract_text_with_ocr(source)
        if ocr_text.strip():
            doc = Document(page_content=ocr_text, metadata={'source': source, "ocr_applied": True})
    else:
        # Fallback: treat as text file
        logger.info(f"\tLoading text file: {source}")
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        doc = Document(page_content=content, metadata={'source': source})
    return doc

# Index documents from URLs and create a retriever for RAG

class VectorStoreManager:
    """
    Vector Store Manager for Legal Document Processing.
    
    This class manages FAISS vector stores with GPU acceleration support,
    handling document indexing, embedding generation, and similarity search
    for legal document retrieval.
    
    Features:
    - GPU-accelerated FAISS indexing with configurable memory allocation
    - Multiple document format support (PDF, web, text)
    - Automatic embedding dimension detection
    - IVF (Inverted File) indexing for large-scale similarity search
    - Persistent storage and loading of vector indexes
    - Memory optimization for resource-constrained environments
    
    Attributes:
        file_path (str): Path to vector store file for persistence
        use_gpu (bool): Whether to use GPU acceleration
        nlist (int): Number of clusters for IVF index
        gpu_memory_mb (int): GPU memory allocation in MB
        gpu_res: FAISS GPU resources object
        embeddings: Embedding model instance
        dim (int): Embedding vector dimensionality
        vector_store: FAISS vector store instance
    """
    
    def __init__(self, file_path=None, nlist=100, gpu_memory_mb=128):
        """
        Initialize VectorStoreManager with configurable GPU and memory settings.
        
        Sets up embedding model, detects vector dimensions, and initializes
        GPU resources.
        
        Args:
            file_path (str, optional): Path to existing vector store file
            nlist (int): Number of clusters for IVF index
            gpu_memory_mb (int): GPU memory allocation in MB
        """
        self.file_path = file_path
        self.nlist = nlist
        self.gpu_memory_mb = gpu_memory_mb
        self.gpu_res = None
        self.batch_size = 128
        
        # Configure embedding model with optimized settings
        self.embeddings_config = {"wait_time": 60}
        self.embeddings = get_embedding_model("mistral-embed", **self.embeddings_config)

        # Detect embedding dimensionality
        test_embedding = self.embeddings.embed_query("Test query to determine embedding dimensionality.")
        self.dim = len(test_embedding)
        logger.info(f"Embedding dimension detected: {self.dim}")


        # Initialize GPU resources with configurable memory allocation
        self.gpu_res = StandardGpuResources()
        self.gpu_res.setTempMemory(self.gpu_memory_mb * 1024 * 1024)
        logger.info(f"GPU resources initialized for FAISS with {self.gpu_memory_mb} MB memory allocation.")
          
        if self.file_path: # Load existing vector store from disk
            self.vector_store = FAISS.load_local(
                self.file_path, self.embeddings, allow_dangerous_deserialization=True
            )
            self._move_to_gpu()
        else: # Create a new vector store
            # Start with 'Flat' and upgrade to IVF when dataset size exceeds threshold
            index = self._create_optimized_index()
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1500,
            chunk_overlap=150, # 10% overlap
            separators=["\n\n", "\n", ".", "!", "?", "。", "！", "？", " "]
        )

    def _train_index(self, embedding_vectors: list[np.ndarray]):
        """
        Train the FAISS index if it requires training (IVF indices).
        Also handles switching from Flat to IVF index when enough vectors are available.
        
        Args:
            embedding_vectors: List of embedding vectors for training
        """
        index = self.vector_store.index
        num_vectors = len(embedding_vectors)

        # Check if we have enough vectors to qualify for IVF
        ivf_index_required = num_vectors + index.ntotal >= self.nlist * 2 
        
        # Upgrade
        if ("Flat" in type(index).__name__ and 
            ivf_index_required):  # Check if we have enough vectors to qualify for IVF
            logger.info(f"Switching to IVF index with {num_vectors + index.ntotal} vectors (>= {self.nlist * 2})")
            self._upgrade_to_ivf_index()
            index = self.vector_store.index
        
            # Check if we need to train the index
            if hasattr(index, "is_trained") and not index.is_trained:
                # Check if we have enough vectors to train IVF index
                if ivf_index_required:
                    logger.info(f"Not enough vectors to train IVF index with {self.nlist} clusters. "
                        f"Need at least {self.nlist} vectors. Deferring training.")
                    return
                
            vectors_np = np.array(embedding_vectors, dtype='float32')
            logger.info(f"Training index with {len(vectors_np)} vectors...")
            
            # Training happens on the current device (GPU or CPU)
            index.train(vectors_np)
            logger.info("Index training completed.")
     
    def _move_to_gpu(self):
        """
        Move the loaded vector store index to GPU.
        Handles different index types that might be loaded from disk.
        """
        if self.gpu_res:
            current_index = self.vector_store.index
            
            # Check if index is already on GPU
            if "Gpu" in type(current_index).__name__:
                logger.info("Index is already on GPU.")
                return
                
            try:
                # Move index to GPU
                gpu_index = index_cpu_to_gpu(self.gpu_res, 0, current_index)
                self.vector_store.index = gpu_index
                logger.info(f"Vector store index moved to GPU after load: {type(gpu_index).__name__}")
            except Exception as e:
                logger.info(f"Failed to move index to GPU (will continue with CPU): {e}")

    def _upgrade_to_ivf_index(self):
        """
        Upgrade from a Flat index to an IVF index when we have enough vectors.
        """
        current_index = self.vector_store.index
        
        # Extract all vectors from current index
        if current_index.ntotal > 0:
            logger.info(f"Extracting {current_index.ntotal} vectors from current index...")
            vectors = np.zeros((current_index.ntotal, self.dim), dtype='float32')
            for i in range(current_index.ntotal):
                vectors[i] = current_index.reconstruct(i)
        else:
            vectors = np.empty((0, self.dim), dtype='float32')
        
        # Create new IVF index
        quantizer = faiss.IndexFlatL2(self.dim)
        new_index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
        
        # Move to GPU if needed
        if self.gpu_res:
            try:
                new_index = index_cpu_to_gpu(self.gpu_res, 0, new_index)
                logger.info("New IVF index moved to GPU.")
            except Exception as e:
                logger.info(f"Failed to move new IVF index to GPU: {e}")
                logger.info("Continuing with CPU-based IVF index.")
                # Don't disable GPU entirely, just use CPU for this index
        
        # Train the new index if we have enough vectors
        if len(vectors) >= self.nlist:
            new_index.train(vectors)
            logger.info(f"New IVF index trained with {len(vectors)} vectors.")
        
        # Add existing vectors to new index
        if len(vectors) > 0:
            new_index.add(vectors)
            logger.info(f"Added {len(vectors)} existing vectors to new IVF index.")
        
        # Replace the index in vector store
        self.vector_store.index = new_index
        logger.info("Successfully upgraded to IVF index.")

    def _index_document(self, content, doc_metadata={'type': "unknown"}):
        """
        Index a single document by splitting it into chunks and adding to vector store.
        
        Args:
            content: Document content to index
            doc_metadata: Metadata for the document
            
        Returns:
            List of chunk UUIDs
        """
        doc_chunks = self.text_splitter.split_text(content)
        chunk_uuids = [str(uuid4()) for _ in range(len(doc_chunks))]
        
        # Clean up chunks to remove any empty strings
        doc_chunks = [chunk for chunk in doc_chunks if chunk.strip()]

        
        # Generate embeddings for all chunks
        vectors = [self.embeddings.embed_query(chunk) for chunk in doc_chunks]

        try:
            # Train the index if necessary (only for IVF indices)
            self._train_index(vectors)
            
            # Add texts with appropriate batch size for GPU performance
            self.vector_store.add_texts(
                texts=doc_chunks,
                metadatas=[doc_metadata] * len(doc_chunks),
                ids=chunk_uuids,
                batch_size=self.batch_size
            )
            logger.info(f"Successfully indexed {len(doc_chunks)} chunks from {doc_metadata.get('type', 'unknown')} document.")
        except Exception as e:
            logger.info(f"Error adding document {doc_metadata.get('type', 'unknown')} to vector store: {e}")
        return chunk_uuids

    def save_vector_store(self, index_path=None):
        if not index_path:
            if not self.file_path:
                timestamp = int(time.time())
                index_path = os.path.join("data/vector_db", f"vector_store_{timestamp}")
                os.makedirs(index_path, exist_ok=True)
            else:
                index_path = self.file_path
        else:
            os.makedirs(index_path, exist_ok=True)

        # Move index to CPU before saving if on GPU
        index_on_gpu = False
        current_index = self.vector_store.index
        
        # Check if index is on GPU by examining its type name
        if "Gpu" in type(current_index).__name__:
            try:
                import faiss
                cpu_index = faiss.index_gpu_to_cpu(current_index)
                self.vector_store.index = cpu_index
                index_on_gpu = True
                logger.info("Moved index to CPU for saving.")
            except Exception as e:
                logger.info(f"Failed to move index to CPU for saving: {e}")

        self.vector_store.save_local(index_path)
        self.file_path = index_path
        logger.info(f"Saved vector store to: {index_path}")

        # Move back to GPU if needed
        if index_on_gpu and self.gpu_res:
            try:
                from faiss import index_cpu_to_gpu
                self.vector_store.index = index_cpu_to_gpu(self.gpu_res, 0, self.vector_store.index)
                logger.info("Restored index to GPU after saving.")
            except Exception as e:
                logger.info(f"Failed to restore index to GPU after saving: {e}")
        return index_path

    def index_file_documents(self, paths, source_type):
        meta_datas = []
        for path in tqdm(paths, desc="Indexing file documents"):
            doc = load_source_with_fallback(path)
            if not doc:
                logger.info(f"Failed to load document from {path}. Skipping.")
                continue
            doc.metadata["type"] = source_type
            doc.metadata["chunk_ids"] = self._index_document(doc.page_content, doc.metadata)
            meta_datas.append(doc.metadata)
        return meta_datas
    
    def index_web_documents(self, urls, source_type):
        web_documents = WebBaseLoader(urls, continue_on_failure=True).load()
        # Process each URL
        meta_datas = []
        for doc in web_documents:
            # Doc metadata set by WebBaseLoader: 
            #   "source": url
            #   "title": soup.find("title").get_text()
            #   "description": soup.find("meta", attrs={"name": "description"}).get("content", "No description found.")
            #   "language": soup.find("html").get("lang", "No language found.")
            doc.metadata["source_type"] = source_type
            doc.metadata["chunk_ids"] = self._index_document(doc.page_content, doc.metadata)
            meta_datas.append(doc.metadata)
        return meta_datas

    def add_texts(self, texts, metadatas=None, ids=None):
        """
        Add texts to the vector store with proper training if needed.
        Optimized for GPU performance when available.
        
        Args:
            texts: List of text strings to add
            metadatas: List of metadata dictionaries (optional)
            ids: List of document IDs (optional)
            
        Returns:
            List of document IDs
        """
        if not ids:
            ids = [str(uuid4()) for _ in range(len(texts))]
        
        # Generate embeddings for training if needed
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        vectors = [self.embeddings.embed_query(text) for text in texts]
        
        # Train the index if necessary
        self._train_index(vectors)
        
        # Add the texts to the vector store
        result = self.vector_store.add_texts(
            texts=texts, metadatas=metadatas, ids=ids, batch_size=self.batch_size
        )
        
        logger.info(f"Successfully added {len(texts)} texts to vector store.")
        return result

    def query_vector_store(self, query, k=5, meta_filter=None):
        if not self.vector_store:
            logger.info("No vector store loaded.")
            return []
        return self.vector_store.similarity_search_with_score(query, k=k, filter=meta_filter)

    def _create_optimized_index(self):
        """
        Create an optimized FAISS index based on expected dataset size and GPU availability.
        Starts with a Flat index and upgrades to IVF when enough vectors are available.
        
        Returns:
            FAISS index optimized for the current configuration
        """
        # Always start with a Flat index for small datasets
        # We'll upgrade to IVF when we have enough vectors (>= nlist * 2)
        index = faiss.IndexFlatL2(self.dim)
        
        if self.gpu_res:
            try:
                # Move to GPU immediately for better performance
                index = index_cpu_to_gpu(self.gpu_res, 0, index)
                logger.info(f"Created Flat index on GPU (will upgrade to IVF when >= {self.nlist * 2} vectors)")
                return index
            except Exception as e:
                logger.info(f"Failed to create GPU index: {e}")
                logger.info("This might be due to GPU memory limitations. Falling back to CPU.")
                self.gpu_res = None
                
        # CPU index
        logger.info(f"Using CPU-based IndexFlatL2 (will upgrade to IVF when >= {self.nlist * 2} vectors)")
        return index

    def get_index_info(self):
        """
        Get information about the current FAISS index configuration.
        
        Returns:
            Dictionary with index information
        """
        if not self.vector_store or not self.vector_store.index:
            return {"status": "No index available"}
            
        index = self.vector_store.index
        info = {
            "index_type": type(index).__name__,
            "dimension": self.dim,
            "total_vectors": index.ntotal,
            "is_trained": getattr(index, "is_trained", True),
            "on_gpu": "Gpu" in type(index).__name__
        }
        
        if hasattr(index, "nlist"):
            info["nlist"] = index.nlist
            
        return info
    
    def print_index_status(self):
        """
        Print current index status and configuration.
        """
        info = self.get_index_info()
        logger.info("=== FAISS Index Status ===")
        for key, value in info.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 26)


    def get_case_vector_store_dict(self):
        """
        Get a dictionary representation of the vector store for case management.
        """
        return self.vector_store.docstore._dict

    def delete_documents(self, uuids_to_delete):
        """
        Delete documents from the vector store by their UUIDs.
        
        Args:
            uuids_to_delete (list): List of UUIDs to delete from the vector store
        """
        if not uuids_to_delete:
            logger.info("No UUIDs provided for deletion.")
            return
        
        logger.info(f"Deleting {len(uuids_to_delete)} documents from vector store...")
        self.vector_store.delete(uuids_to_delete)
        logger.info("Deletion completed.")
