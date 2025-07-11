"""
Database Manager for Legal RAG Pipeline
=======================================

This module provides comprehensive database management functionality for the
Legal RAG Pipeline, handling both SQL database operations and vector store
management for semantic document search.

The DatabaseManager class serves as the central orchestrator for:
- Client and case management
- Document upload and metadata tracking
- Vector database creation and querying
- Conversation history management
- Integration between SQL and vector databases


"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .sql_classes import *
from .vector_database import VectorStoreManager
from datetime import datetime
from collections import deque

class DatabaseManager:
    """
    Comprehensive database manager for legal case management and document retrieval.
    
    This class handles all database operations including SQL database management
    for structured data (clients, cases, metadata) and vector database management
    for semantic document search and retrieval.
    
    The manager maintains state for the current active client and case, manages
    conversation history, and coordinates between different storage systems.
    
    Attributes:
        root_vector_db_dir (str): Root directory for vector database storage
        engine: SQLAlchemy database engine
        session: SQLAlchemy database session
        case_vector_store_manager: Vector store manager for current case
        conversation_history: Deque storing recent conversation history
        client: Currently active client object
        case: Currently active case object
    """

    def __init__(self, root_vector_db_dir, db_dir):
        """
        Initialize the database manager with storage paths and connections.
        
        Creates database engine, initializes tables, and sets up vector storage
        directory structure. Prepares for client and case management.
        
        Args:
            root_vector_db_dir (str): Directory path for storing vector databases
            db_dir (str): Directory path containing SQLite database
        """
        # Configure storage directories
        self.root_vector_db_dir = root_vector_db_dir
        
        # Set up SQLAlchemy engine and session for structured data
        db_path = os.path.join(db_dir, "database.db")
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)  # Create tables if they don't exist
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Initialize vector store management
        self.case_vector_store_manager = None
        
        # Initialize conversation history with limited size for memory efficiency
        self.conversation_history = deque(maxlen=25)
        
        # Initialize current client and case state
        self.client = None
        self.case = None

    def initialize_from_case(self, case: Case):
        """
        Initialize the database manager from an existing case.
        
        Sets up the vector store manager for the specified case and loads
        relevant conversation history. This method is called when switching
        between cases or loading a case for the first time.
        
        Args:
            case (Case): Case object to initialize from
        """
        # Set the current case
        self.case = case
        # Set the current client based on the case
        self.client = self.get_client_by_id(case.client_id)  
        # Load the vector store for the case
        store = self.session.query(VectorStore).filter(VectorStore.case_id == self.case.id).first()
        self.case_vector_store_manager = VectorStoreManager(file_path=store.file_path if store else None)
        if not store:
            self.add_case_vector_store() # Create a new vector store if it doesn't exist, save it to the database and file system
        
        # Load chat history for the case into memory
        chats = self.session.query(ChatHistory).filter_by(case_id=self.case.id).order_by(ChatHistory.timestamp).all()
        self.conversation_history.extend((chat.user_prompt, chat.assistant_response) for chat in chats)

    # Data getters and setters

    def get_client_by_id(self, client_id):
        """
        Get a client by their ID.
        
        Args:
            client_id: ID of the client to retrieve
            
        Returns:
            Client object if found, None otherwise
        """
        return self.session.query(Client).filter_by(id=client_id).first()

    def get_cases_by_client_id(self, client_id):
        """
        Get all cases associated with a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            List of Case objects
        """
        return self.session.query(Case).filter_by(client_id=client_id).all()

    def get_clients_by_case_id(self, case_id):
        """
        Get the client associated with a case.
        
        Args:
            case_id: ID of the case
            
        Returns:
            Client object if found, None otherwise
        """
        return self.session.query(Client).join(Case).filter(Case.id == case_id).first()

    def get_cases_by_name(self, name):
        """
        Search for cases by name.
        
        Args:
            name: Case name to search for (partial match)
            
        Returns:
            List of matching Case objects
        """
        return self.session.query(Case).filter(Case.name.like(f"%{name}%")).all()

    def get_case_by_id(self, case_id):
        """
        Get a case by its ID.
        
        Args:
            case_id: ID of the case to retrieve
            
        Returns:
            Case object if found, None otherwise
        """
        return self.session.query(Case).filter_by(id=case_id).first()

    def get_clients_by_name(self, name):
        """
        Get clients by name (partial match).
        
        Args:
            name: Name or partial name to search for
            
        Returns:
            List of matching Client objects
        """
        if not self.case:
            raise ValueError("Case must be set before retrieving clients by name.")
        return self.session.query(Client).join(Case).filter(
            Case.id == self.case.id,
            Client.name.like(f"%{name}%")
        ).all()

    def get_uploaded_documents(self):
        """
        Get all uploaded documents for the current case.
        
        Returns:
            List of SourceDocument objects
        """
        if not self.case:
            return []
        return self.session.query(SourceDocument).filter_by(case_id=self.case.id).all()

    def get_chat_history(self):
        """
        Get the chat history for the current case.
        
        Returns:
            List of ChatHistory objects
        """
        if not self.case:
            return []
        return self.session.query(ChatHistory).filter_by(case_id=self.case.id).all()

    def get_case_vector_store(self):
        """
        Get the vector store for the current case.
        
        Returns:
            VectorStore object if found, None otherwise
        """
        if not self.case:
            return None
        return self.session.query(VectorStore).filter_by(case_id=self.case.id).first()
    
    def get_vector_store_by_id(self, vector_store_id):
        # Retrieve a vector store by its ID
        return self.session.query(VectorStore).filter(VectorStore.id == vector_store_id).first()

    def get_uploaded_documents_by_name(self, names):
        # Retrieve source documents whose names match the search term(s) (case-insensitive)
        if not self.case:
            raise ValueError("Case must be set before retrieving documents by name.")
        if isinstance(names, str):
            names = [names]
        return self.session.query(SourceDocument).filter(
            SourceDocument.case_id == self.case.id,
            SourceDocument.source_name.in_(names)
        ).all()

    def add_case_client(self, case: Case, client: Client=None, **kwargs):
        # Add a new client (from object or kwargs) and link to the current case
        if client is None:
            client = Client(**kwargs)
            self.session.add(client)
            self.session.commit()
            self.session.refresh(client)
        else:
            if not self.session.object_session(client):
                self.session.add(client)
                self.session.commit()
                self.session.refresh(client)
        case.client_id = client.id
        self.session.commit()
        self.session.refresh(case)
        return client

    def add_case(self, **kwargs):
        # Create a new case for a client
        case = Case(**kwargs)
        self.session.add(case)
        self.session.commit()
        self.session.refresh(case)
        return case

    def add_case_vector_store(self):
        # Create a new vector store for the current case
        if not self.case or not self.case_vector_store_manager:
            raise ValueError("Case must be set before creating a vector store.")
        if not os.path.exists(self.root_vector_db_dir):
            os.makedirs(self.root_vector_db_dir)
        data_dir = f"vector_store_{self.case.id}"
        vector_store_dir = os.path.join(self.root_vector_db_dir, data_dir)
        os.makedirs(vector_store_dir, exist_ok=True)
        self.case_vector_store_manager.save_vector_store(vector_store_dir)
        store = VectorStore(
            case_id=self.case.id,
            file_path=vector_store_dir,
        )
        self.session.add(store)
        self.session.commit()
        self.session.refresh(store)
        return store

    def add_case_chat_history(self, user_prompt, assistant_response):
        # Add a chat interaction to the database and cache
        if not self.case or not self.case_vector_store_manager:
            raise ValueError("Case and vector store manager must be set before adding chat history.")
        chunk_ids = self.case_vector_store_manager._index_document(user_prompt, {'type': "chat"})

        chat = ChatHistory(
            case_id=self.case.id,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            chunk_ids=chunk_ids,
        )
        self.session.add(chat)
        self.session.commit()
        self.session.refresh(chat)
        # Add to in-memory cache
        self.conversation_history.append((user_prompt, assistant_response))
        return chat

    def _add_source_documents(self, meta_datas):
        # Helper method to add source documents to the database and vector store
        if not self.case:
            raise ValueError("Case must be set before adding source documents.")
        documents = []
        for meta in meta_datas:
            doc = SourceDocument(
                case_id=self.case.id,
                source_name=meta.get('source_name', ""),
                title=meta.get('title', ""),
                description=meta.get('description', ""),
                language=meta.get('language', ""),
                chunk_ids=meta.get('chunk_ids', []),
            )
            documents.append(doc)
        self.session.add_all(documents)
        self.session.commit()
        return documents

    def add_web_documents(self, urls, source_type="uploaded"):
        # Add documents from web URLs to the vector store
        doc_meta_datas = self.case_vector_store_manager.index_web_documents(urls, source_type)
        document_records = self._add_source_documents(doc_meta_datas)
        return document_records

    def add_file_documents(self, file_paths, source_type="uploaded"):
        # Add documents from file paths to the vector store
        doc_meta_datas = self.case_vector_store_manager.index_file_documents(file_paths, source_type)
        document_records = self._add_source_documents(doc_meta_datas)
        return document_records

    def query_documents(self, query, k=5, meta_filter=None):
        # Query the vector store for relevant documents
        results = self.case_vector_store_manager.query_vector_store(query, k=k, meta_filter=meta_filter)
        return results

    def delete_documents(self, ids):
        # Delete documents from the vector store and update or delete source documents in the database
        uuids_to_delete = []
        docs = self.session.query(SourceDocument).filter(SourceDocument.id.in_(ids)).all()
        for doc in docs:
            uuids_to_delete.extend(doc.chunk_ids)
            self.session.delete(doc)
        self.session.commit()
        return self.case_vector_store_manager.delete_documents(uuids_to_delete)

    def save_all(self):
        # Commit all changes to the database and save the vector store
        self.session.commit()
        vector_path = os.path.join(self.root_vector_db_dir, f"vector_store_{self.case.id}")
        return self.case_vector_store_manager.save_vector_store(vector_path)