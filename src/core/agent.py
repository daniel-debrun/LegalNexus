"""
Legal RAG Agent - Core Orchestrator

This module contains the main LegalRagAgent class that serves as the central
coordinator for the Legal RAG Pipeline system. It integrates language models,
vector databases, and legal research capabilities to provide intelligent
legal assistance.

Key Features:
- Multi-model LLM support (Mistral, OpenAI, Ollama)
- RAG pipeline for contextual legal advice
- CanLII integration for Canadian legal research
- Document upload and semantic search
- Case and client management
- Conversation history tracking

Author: Legal RAG Pipeline Team
Version: 1.0.0
"""

from dotenv import load_dotenv
import re
import os
import json
from datetime import datetime
from collections import deque
from typing import List, Optional, Dict, Any
import logging

# Core RAG components
from src.core.rag_pipeline import RAGPipeline, build_workflow
from src.storage.database_manager import DatabaseManager

# LLM providers
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama

# Legal-specific prompt tools
from src.generation.rag_prompt_tools import (
    ChatSummarizer,
    PassiveLegalAdvice,
    CaseSimilaritySearch,
    LegislationSimilaritySearch,
    CanliiQueryWriter,
)

# Utilities
from src.utils.canlii_search import search_canlii
from src.generation.model_provider import get_llm_model

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalRagAgent:
    """
    Legal RAG Agent - Main orchestrator for the legal assistance system.
    
    This class coordinates interactions between the database manager, 
    RAG pipeline, and language models to provide intelligent legal
    information and advice based on client cases, documents, and
    legal knowledge databases.
    
    The agent supports:
    - Case and client management
    - Document upload and semantic search
    - Legal research via CanLII integration
    - Contextual question answering
    - Conversation history tracking
    - Multi-model LLM support
    
    Attributes:
        database_manager (DatabaseManager): Handles data persistence and retrieval
        rag_pipeline (RAGPipeline): Manages document retrieval and answer generation
        llm: Current language model instance
        graph: Workflow graph for RAG operations
    """

    def __init__(self, database_manager: DatabaseManager, model_name: str = "mistral-large-2411"):
        """
        Initialize the Legal RAG Agent.
        
        Args:
            database_manager: Database manager instance for data operations
            model_name: Name of the language model to use (default: mistral-large-2411)
            
        Raises:
            ValueError: If required environment variables are missing
        """
        self.database_manager = database_manager
        self.rag_pipeline = None
        self.graph = None
        
        # Validate required environment variables
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key and 'mistral' in model_name.lower():
            raise ValueError(
                "MISTRAL_API_KEY environment variable is required for Mistral models. "
                "Please set it in your .env file."
            )

        # Configure language model
        self.llm_config = {
            "temperature": 0.0,        # Deterministic responses for legal advice
            "num_ctx": 100000,         # Large context window for legal documents
            "extract_reasoning": False  # Focus on final answers
        }
        
        self.llm = get_llm_model(model_name, True, **self.llm_config)

        # Initialize RAG pipeline and workflow
        self.rag_pipeline = RAGPipeline(llm=self.llm)
        self.graph = build_workflow(self.rag_pipeline)
        
        # Initialize chat summarizer for conversation management
        self.chat_summarizer = ChatSummarizer(self.llm)
        
        logger.info(f"Legal RAG Agent initialized with {model_name}")
    
    def set_llm(self, model_name: str) -> bool:
        """
        Update the language model used by the agent.
        
        Args:
            model_name: Name of the new language model
            
        Returns:
            bool: True if model was successfully updated, False otherwise
        """
        try:
            self.llm = get_llm_model(model_name, True, **self.llm_config)
            self.rag_pipeline.llm = self.llm
            self.graph = build_workflow(self.rag_pipeline)
            logger.info(f"Language model updated to {model_name}")
            return True
        except Exception as e:
            logger.info(f"Failed to update language model: {str(e)}")
            return False

    def _get_object_handler(self, search_type, query):
        """
        Search for an existing client or case by name or ID.
        If not found, returns None.
        """
        if not query:
            query = input(f"Enter name to search for an existing {search_type}, or press Enter to create a new {search_type}: ").strip()

        if search_type == "client":
            matches = self.database_manager.get_client_by_id(query)
        elif search_type == "case":
            matches = self.database_manager.get_cases_by_name(query)
        else:
            logger.info(f"Unknown search_type: {search_type}")
            return None

    # 
    # WEB-SAFE SEARCH METHODS
    # 
    
    def get_case_by_name_web_safe(self, case_name: str) -> Optional[Any]:
        """
        Web-safe version of case search that doesn't use input() calls.
        
        This method is designed for use by web APIs where interactive
        input is not possible.
        
        Args:
            case_name: Name or ID of the case to find
            
        Returns:
            Case object if found, None otherwise
        """
        if not case_name:
            return None
            
        # Try to get case by ID first (if it's a number)
        if case_name.isdigit():
            case = self.database_manager.get_case_by_id(case_name)
            if case:
                return case
        
        # Search by name
        matches = self.database_manager.get_cases_by_name(case_name)
        if matches:
            return matches[0]  # Return first match for web safety
            
        return None

    def get_client_by_name_web_safe(self, client_name: str) -> Optional[Any]:
        """
        Web-safe version of client search that doesn't use input() calls.
        
        This method is designed for use by web APIs where interactive
        input is not possible.
        
        Args:
            client_name: Name or ID of the client to find
            
        Returns:
            Client object if found, None otherwise
        """
        if not client_name:
            return None
            
        # Try to get client by ID first (if it's a number)
        if client_name.isdigit():
            client = self.database_manager.get_client_by_id(client_name)
            if client:
                return client
        
        # Search by name
        matches = self.database_manager.get_clients_by_name(client_name)
        if matches:
            return matches[0]  # Return first match for web safety
            
        return None

    def _new_object_from_json(self, search_type: str, json_path: Optional[str] = None) -> Optional[Any]:
        """
        Create a new client or case object from JSON file data.
        
        Args:
            search_type: Type of object to create ("client" or "case")
            json_path: Path to JSON file containing object data
            
        Returns:
            Newly created client or case object, or None if creation fails
        """
        if not json_path or not os.path.exists(json_path):
            logger.info(f"JSON file not found: {json_path}")
            return None
            
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {search_type} data from {json_path}")
            
            if search_type == "client":
                return self.database_manager.add_client(**data)
            elif search_type == "case":
                if self.database_manager.client:
                    return self.database_manager.add_case(
                        client_id=self.database_manager.client.id, 
                        **data
                    )
                else:
                    return self.database_manager.add_case(**data)
            else:
                logger.info(f"Unknown search_type: {search_type}")
                return None
                
        except Exception as e:
            logger.info(f"Error creating {search_type} from JSON: {str(e)}")
            return None


    # 
    # DOCUMENT MANAGEMENT METHODS  
    # 
    def _handle_documents(self, message: str) -> str:
        """
        Handle document processing commands embedded in messages.
        
        Args:
            message: Input message that may contain document commands
            
        Returns:
            Response message about document processing
        """
        if isinstance(message, str) and "UPLOAD_DOCUMENTS" in message:
            file_path = os.path.join(os.getcwd(), 'srcdoc.pdf')
            logger.info(f"Uploading document from {file_path}")
            self.upload_documents(doc_files=[file_path])
            return "Document uploaded and processed successfully."
        return message

    def validate_urls(self, urls: List[str]) -> List[str]:
        """
        Validate and filter URL list for document upload.
        
        Args:
            urls: List of URLs to validate
            
        Returns:
            List of valid URLs
        """
        valid_urls = []
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        for url in urls:
            if url_pattern.match(url):
                valid_urls.append(url)
            else:
                logger.info(f"Invalid URL skipped: {url}")
                
        return valid_urls

    def validate_file_paths(self, file_paths: List[str]) -> List[str]:
        """
        Validate and filter file path list for document upload.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            List of valid file paths that exist
        """
        valid_paths = []
        
        for path in file_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.info(f"File not found, skipped: {path}")
                
        return valid_paths

    def upload_documents(self, web_urls: List[str] = None, doc_files: List[str] = None) -> int:
        """
        Upload and index documents from web URLs and local files.
        
        This method processes documents through the vector store for
        semantic search and retrieval during legal consultations.
        
        Args:
            web_urls: List of web URLs to scrape and index
            doc_files: List of local file paths to index
            
        Returns:
            Number of documents successfully uploaded
        """
        web_urls = web_urls or []
        doc_files = doc_files or []
        
        # Validate inputs
        web_urls = self.validate_urls(web_urls)
        doc_files = self.validate_file_paths(doc_files)
        
        documents_added = 0
        
        try:
            # Index web pages if any
            if web_urls:
                logger.info(f"Indexing {len(web_urls)} web pages...")
                web_results = self.database_manager.add_web_documents(web_urls)
                documents_added += len(web_results)
                logger.info(f"Indexed {len(web_results)} web documents")
            
            # Index local files if any
            if doc_files:
                logger.info(f"Indexing {len(doc_files)} local files...")
                file_results = self.database_manager.add_file_documents(doc_files)
                documents_added += len(file_results)
                logger.info(f"Indexed {len(file_results)} file documents")
            
            # Save the updated vector store
            if documents_added > 0:
                self.database_manager.case_vector_store_manager.save_vector_store()
                logger.info(f"Successfully uploaded and indexed {documents_added} documents")
            else:
                logger.info("No valid documents found to upload")
                
        except Exception as e:
            logger.info(f"Error uploading documents: {str(e)}")
            
        return documents_added


    # 
    # RAG QUESTION ANSWERING METHODS
    # 

    def ask_rag_question(self, question: str, uuids_for_retrieval: List[str] = None, 
                        included_web_pages: List[str] = None) -> Optional[Any]:
        """
        Process a legal question through the RAG pipeline.
        
        This method orchestrates the entire question-answering process,
        including document retrieval, context preparation, and answer generation.
        
        Args:
            question: The user's legal question
            uuids_for_retrieval: Specific document UUIDs to include in retrieval
            included_web_pages: Web pages to include in context
            
        Returns:
            Chat interaction object with question and answer, or None if failed
        """
        uuids_for_retrieval = uuids_for_retrieval or []
        included_web_pages = included_web_pages or []
        
        try:
            # Prepare conversation history context
            if not self.database_manager.conversation_history:
                history_text = "This is the first question in this conversation."
            else:
                # Use last 3 interactions for context
                recent_history = list(self.database_manager.conversation_history)[-3:]
                history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in recent_history])
            
            # Prepare client context
            client_notes = ""
            if self.database_manager.client and hasattr(self.database_manager.client, 'notes'):
                client_notes = f"Client: {self.database_manager.client.notes or ''}"
            
            # Prepare case context
            case_details = ""
            if self.database_manager.case:
                case_details = (
                    f"Case: {self.database_manager.case.name}, "
                    f"Type: {getattr(self.database_manager.case, 'case_type', 'Unknown')}, "
                    f"Jurisdiction: {self.database_manager.case.jurisdiction_code}"
                )
            
            # Prepare state for RAG pipeline
            state = {
                "question": question,
                "case_details": case_details,
                "client_notes": client_notes,
                "conversation_history": history_text,
                "documents_to_retrieve": uuids_for_retrieval,
                "web_pages": included_web_pages,
            }
            
            # Run the RAG pipeline
            logger.info(f"Processing question: {question[:50]}...")
            result = self.graph.invoke(state)
            answer = result.get("generation", "Sorry, I couldn't generate an answer.")
            
            # Save the interaction to database if case exists
            if self.database_manager.case:
                chat_interaction = self.database_manager.add_case_chat_history(question, answer)
                logger.info("Question processed and saved to chat history")
                return chat_interaction
            else:
                logger.info("No case loaded - conversation not saved")
                return None
                
        except Exception as e:
            logger.info(f"Error processing question: {str(e)}")
            return None

    def execute_actions(self, question: str) -> Optional[Any]:
        """
        Execute actions based on the user's question.
        
        This method handles special commands and routes questions to
        appropriate processing methods.
        
        Args:
            question: User's input question or command
            
        Returns:
            Chat interaction object or None if processing failed
        """
        # Handle document upload commands
        processed_message = self._handle_documents(question)
        if processed_message != question:
            # Document command was processed
            return None
        
        # Check for special commands
        commands = self.match_commands(question)
        if commands:
            logger.info(f"Executing command: {commands}")
            # Handle commands here if needed
        
        # Process as regular RAG question
        return self.ask_rag_question(question)

    def match_commands(self, question: str) -> Optional[str]:
        """
        Match question against known command patterns.
        
        Args:
            question: Input question to check for commands
            
        Returns:
            Matched command string or None
        """
        # Define command patterns
        patterns = {
            r'#summary': 'GENERATE_SUMMARY',
            r'#doc': 'DOCUMENT_SEARCH',
            r'#web': 'WEB_SEARCH',
            r'#save': 'SAVE_SESSION',
        }
        
        for pattern, command in patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                return command
        
        return None

    def clear_conversation_history(self) -> None:
        """
        Clear the current conversation history.
        
        This removes all cached conversation history but does not
        affect the persistent database records.
        """
        self.database_manager.conversation_history.clear()
        logger.info("Conversation history cleared")

    def format_conversation_history(self) -> str:
        """
        Format conversation history for display or context.
        
        Returns:
            Formatted string of recent conversation history
        """
        if not self.database_manager.conversation_history:
            return "No conversation history available."
        
        formatted_history = []
        for i, (question, answer) in enumerate(self.database_manager.conversation_history, 1):
            formatted_history.append(f"Q{i}: {question}")
            formatted_history.append(f"A{i}: {answer}")
            formatted_history.append("")  # Empty line for readability
        
        return "\n".join(formatted_history)


    # 
    # LEGAL RESEARCH AND ADVICE METHODS
    # 
    def generate_passive_legal_information(self) -> Dict[str, Any]:
        """
        Generate contextual legal information and resources.
        
        This method provides general legal guidance that may be relevant
        to the current case and conversation context. It combines AI-generated
        advice with relevant web resources.
        
        Returns:
            Dictionary containing advice, resources, and search results
        """
        try:
            # Generate AI advice based on conversation history
            passive_legal_generator = PassiveLegalAdvice(self.llm)
            formatted_conversation_history = self.format_conversation_history()
            
            advice_result = passive_legal_generator.invoke({
                'conversation_history': formatted_conversation_history
            })
            advice = advice_result.content if hasattr(advice_result, 'content') else str(advice_result)
            
            # Search for relevant web resources
            query = "legal information general guidance"
            if self.database_manager.case and hasattr(self.database_manager.case, 'case_type'):
                case_type = getattr(self.database_manager.case, 'case_type', '')
                if case_type:
                    query = f"legal information {case_type}"
            
            # Perform web search for relevant resources
            links = []
            titles = []

            search_results = self.rag_pipeline.web_search({"question": query})
            for doc in search_results.get("documents", []):
                if hasattr(doc, 'metadata'):
                    url = doc.metadata.get("url", "")
                    title = doc.metadata.get("title", "Legal Resource")
                    if url:
                        links.append(url)
                        titles.append(title)

            return {
                "advice": advice,
                "links": links,
                "link_titles": titles
            }
            
        except Exception as e:
            logger.info(f"Error generating passive legal information: {str(e)}")
            return {
                "advice": "Please consult with a qualified legal professional for specific legal advice.",
                "links": ["https://www.canlii.org/", "https://laws-lois.justice.gc.ca/", "https://www.justice.gc.ca/"],
                "link_titles": [
                    "CanLII - Canadian Legal Information Institute", 
                    "Justice Laws Website - Government of Canada",
                    "Department of Justice Canada"]
            }

    def set_legal_references(self, top_k: Optional[int] = 10) -> None:
        """
        Set up legal references for the current case by searching CanLII.
        
        This method automatically searches Canadian legal databases for
        relevant case law and legislation based on the current case details.
        The AI evaluates and filters the most relevant legal precedents.
        
        Args:
            top_k: Maximum number of references to retrieve (default: use system default)
        """
        if not self.database_manager.case:
            logger.info("No case is currently selected. Cannot set legal references.")
            return
            
        try:
            # Prepare case details for search
            case_details = self.database_manager.case.name
            if hasattr(self.database_manager.case, 'notes') and self.database_manager.case.notes:
                case_details += "\n" + self.database_manager.case.notes

            # Generate search query using AI
            query_writer = CanliiQueryWriter(self.llm)
            query_result = query_writer.invoke({"case_details": case_details})
            search_query = query_result.query if hasattr(query_result, 'query') else str(query_result)
            
            logger.info(f"Searching CanLII with query: {search_query}")

            # Search for both legislation and case law
            for search_type in ['LEGISLATION', 'CASE']:
                try:
                    logger.info(f"Searching for relevant {search_type.lower()}...")
                    results = search_canlii(
                        jurisdiction_code=self.database_manager.case.jurisdiction_code,
                        search_term=search_query,
                        search_type=search_type,
                        top_k=top_k
                    )

                    if results:
                        logger.info(f"Found {len(results)} {search_type.lower()} results")
                        
                        # Use AI to evaluate relevance
                        prompt = {
                            "case_details": case_details,
                            "reference_list": [r.get('description', r.get('title', '')) for r in results]
                        }
                        
                        if search_type == 'CASE':
                            logger.info("Evaluating and filtering relevant cases...")
                            relevance_evaluator = CaseSimilaritySearch(llm=self.llm)
                        else:
                            logger.info("Evaluating and filtering relevant legislation...")
                            relevance_evaluator = LegislationSimilaritySearch(llm=self.llm)
                        relevant_items = relevance_evaluator.invoke(prompt=prompt).items
                                                
                        # Index relevant references for retrieval
                        if relevant_items:
                            reference_urls = []
                            for i, item in enumerate(relevant_items):
                                if item.binary_score == '1':
                                    url = results[i].get('path', '')
                                    if url:
                                        reference_urls.append(url)
                            reference_urls = [results[0].get('path', '')]
                            if reference_urls:
                                logger.info(f"Indexing {len(reference_urls)} reference documents...")
                                self.database_manager.add_file_documents(
                                    reference_urls, 
                                    source_type="legal_reference"
                                )
                    else:
                        logger.info(f"No {search_type.lower()} results found")
                        
                except Exception as search_error:
                    logger.info(f"Error searching {search_type}: {search_error}")
                    
            logger.info("Legal references setup completed")
            
        except Exception as e:
            logger.info(f"Error setting legal references: {str(e)}")

    def get_chat_summary(self) -> str:
        """
        Generate a summary of the current chat conversation.
        
        Returns:
            Formatted summary of the conversation history
        """
        if not self.database_manager.conversation_history:
            return "No conversation history available to summarize."
        
        try:
            # Format conversation for summarization
            chat_text = "\n".join([
                f"Q: {q}\nA: {a}" 
                for q, a in self.database_manager.conversation_history
            ])
            
            # Generate summary using AI
            summary_result = self.chat_summarizer.invoke(chat_text)
            summary = getattr(summary_result, 'summary', str(summary_result))
            
            return f"## Chat Summary\n\n{summary}"
            
        except Exception as e:
            logger.info(f"Error generating chat summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
