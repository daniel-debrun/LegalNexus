"""
SQL Database Models for Legal RAG Pipeline
==========================================

This module defines SQLAlchemy ORM models for the Legal RAG Pipeline database.
These models represent the core entities: clients, cases, documents, chat history,
and vector stores.

The database schema supports:
- Client management with personal details and case associations
- Legal case tracking with jurisdiction and status information  
- Document storage and metadata management
- Chat conversation history
- Vector store references for semantic search


"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from sqlalchemy.orm import declarative_base
Base = declarative_base()


class Client(Base):
    """
    Client model for storing client information and personal details.
    
    This model represents legal clients with their contact information,
    personal details, and relationships to their legal cases. Client
    details are stored as JSON to allow flexible schema evolution.
    
    Attributes:
        id (int): Primary key identifier
        created_at (datetime): Record creation timestamp
        last_updated (datetime): Last modification timestamp
        client_details (dict): JSON structure containing:
            - email: Client email address
            - phone: Phone number
            - address: Mailing address
            - date_of_birth: Birth date (YYYY-MM-DD)
            - gender: Gender identification
            - occupation: Professional occupation
        name (str): Client full name
        notes (str): Additional notes about the client
        cases (relationship): Associated legal cases
    """
    __tablename__ = 'clients'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Client information
    client_details = Column(JSON, default={})
    name = Column(String, default='')
    notes = Column(String, default='')
    
    # Relationships
    cases = relationship("Case", back_populates="client")
    
    # JSON Structure Documentation:
    # client_details = {
    #     "email": "client@example.com",
    #     "phone": "+1234567890", 
    #     "address": "123 Main St, City, Country",
    #     "date_of_birth": "YYYY-MM-DD",
    #     "gender": "Male/Female/Other",
    #     "occupation": "Occupation string"
    # }
    
    @property
    def email(self):
        """Get client email from JSON details."""
        return self.client_details.get('email', '') if self.client_details else ''
    
    @property
    def phone(self):
        """Get client phone from JSON details."""
        return self.client_details.get('phone', '') if self.client_details else ''
    
    @property
    def address(self):
        """Get client address from JSON details."""
        return self.client_details.get('address', '') if self.client_details else ''
    
    @property
    def date_of_birth(self):
        """Get client date of birth from JSON details."""
        return self.client_details.get('date_of_birth', '') if self.client_details else ''
    
    @property
    def gender(self):
        """Get client gender from JSON details."""
        return self.client_details.get('gender', '') if self.client_details else ''
    
    
    @property
    def occupation(self):
        """Get client occupation from JSON details."""
        return self.client_details.get('occupation', '') if self.client_details else ''


class Case(Base):
    """
    Case model for legal cases and proceedings.
    
    This model represents individual legal cases with their associated
    metadata, jurisdiction information, and relationships to clients,
    documents, and conversation history.
    
    Attributes:
        id (int): Primary key identifier
        client_id (int): Foreign key to associated client
        name (str): Case name or identifier
        jurisdiction_code (str): Legal jurisdiction code (e.g., 'ON', 'BC')
        court_level (str): Court level (trial, appellate, supreme)
        legal_issue (str): Primary legal issue category
        notes (str): Detailed case notes and observations
        case_type (str): Type of case (Civil, Criminal, Family, etc.)
        case_status (str): Current status (Open, Closed, Pending, etc.)
        created_at (datetime): Case creation timestamp
        client (relationship): Associated client record
        vector_store (relationship): Associated vector database entries
        uploaded_documents (relationship): Associated document records
        chat_history (relationship): Associated conversation history
    """
    __tablename__ = 'cases'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(Integer, ForeignKey('clients.id'))
    
    # Case details
    name = Column(String, default='')
    jurisdiction_code = Column(String, default='')
    court_level = Column(String, default='')
    legal_issue = Column(String, default='')
    notes = Column(Text, default='')
    case_type = Column(String, default='')
    case_status = Column(String, default='')
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships with cascade deletion
    client = relationship("Client", back_populates="cases")
    vector_store = relationship("VectorStore", back_populates="case", cascade="all, delete-orphan")
    uploaded_documents = relationship("SourceDocument", back_populates="case", cascade="all, delete-orphan") 
    chat_history = relationship("ChatHistory", back_populates="case", cascade="all, delete-orphan")


class VectorStore(Base):
    """
    VectorStore model for managing vector database file references.
    
    This model tracks the file paths and metadata for vector databases
    associated with specific cases. Each case can have multiple vector
    stores for different document types or purposes.
    
    Attributes:
        id (int): Primary key identifier
        type (str): Type of vector store (Case, Legal_Reference, etc.)
        description (str): Human-readable description of store contents
        case_id (int): Foreign key to associated case
        file_path (str): File system path to vector database files
        created_at (datetime): Creation timestamp
        case (relationship): Associated case record
    """
    __tablename__ = 'vector_store'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Vector store details
    type = Column(String, default='Case')
    description = Column(Text, default='')
    case_id = Column(Integer, ForeignKey('cases.id'))
    file_path = Column(String, default='')
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="vector_store")


class SourceDocument(Base):
    """
    SourceDocument model for tracking uploaded document metadata.
    
    This model maintains metadata about documents that have been uploaded
    and processed by the system. It tracks the original source, processing
    details, and chunk identifiers for retrieval.
    
    Attributes:
        id (int): Primary key identifier
        case_id (int): Foreign key to associated case
        source_name (str): Original document name or URL
        title (str): Document title or identifier
        description (str): Detailed description of document contents
        language (str): Document language code
        chunk_ids (list): JSON array of vector database chunk identifiers
        uploaded_at (datetime): Upload timestamp
        case (relationship): Associated case record
    """
    __tablename__ = 'uploaded_documents'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey('cases.id'))
    
    # Document metadata
    source_name = Column(String, default='')
    title = Column(String, default='')
    description = Column(Text, default='')
    language = Column(String, default='')
    chunk_ids = Column(JSON, default=[])
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    case = relationship("Case", back_populates="uploaded_documents")


class ChatHistory(Base):
    """
    ChatHistory model for storing conversation records.
    
    This model maintains a complete record of all interactions between
    users and the AI assistant for each case. It enables conversation
    context and case history tracking.
    
    Attributes:
        id (int): Primary key identifier
        case_id (int): Foreign key to associated case
        user_prompt (str): User's question or input
        assistant_response (str): AI assistant's response
        timestamp (datetime): Interaction timestamp
        chunk_ids (list): JSON array of document chunks used in response
        case (relationship): Associated case record
    """
    __tablename__ = 'chat_history'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey('cases.id'))
    
    # Conversation content
    user_prompt = Column(Text, default='')
    assistant_response = Column(Text, default='')
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    chunk_ids = Column(JSON, default=[])
    
    # Relationships
    case = relationship("Case", back_populates="chat_history")
