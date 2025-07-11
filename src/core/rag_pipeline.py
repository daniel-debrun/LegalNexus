"""
RAG Pipeline for Legal Document Processing and Question Answering
================================================================

This module implements a comprehensive Retrieval-Augmented Generation (RAG)
pipeline specifically designed for legal document processing and intelligent
question answering in legal contexts.

The pipeline orchestrates:
- Document retrieval from vector databases
- Question routing to appropriate data sources
- Document relevance assessment and filtering
- Web search integration for external legal resources
- Answer generation with hallucination detection
- Quality control and validation mechanisms

The workflow is implemented as a state machine using LangGraph, providing
robust error handling and flow control for complex legal query processing.


"""

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

import operator
from typing_extensions import TypedDict
from typing import List, Annotated
from langgraph.graph import StateGraph, END
from src.storage.database_manager import DatabaseManager
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.generation.rag_prompt_tools import (
    RagQuestionRouter,
    AnswerGrader,
    HallucinationGrader,
    QuestionReply,
    DocumentGrader,
    NoContextQuestionReply
)

# Set up logging
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """Custom timeout exception for LLM calls that exceed time limits."""
    pass


def with_timeout_executor(func, timeout_seconds=100):
    """
    Execute function with timeout protection using ThreadPoolExecutor.
    
    This decorator provides timeout protection for LLM calls that might hang
    or take excessive time, ensuring the system remains responsive.
    
    Args:
        func: Function to execute with timeout protection
        timeout_seconds (int): Maximum execution time in seconds
        
    Returns:
        Function wrapper with timeout protection
        
    Raises:
        TimeoutError: If function execution exceeds timeout limit
    """
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FuturesTimeoutError:
                logger.info(f"---TIMEOUT: Function call exceeded {timeout_seconds} seconds---")
                # Attempt to kill the thread and release the LLM resource
                try:
                    thread = threading.current_thread()
                    if thread.is_alive():
                        logger.info("---KILLING THREAD DUE TO TIMEOUT---")
                        raise TimeoutError(f"Function call exceeded {timeout_seconds} seconds and thread was terminated")
                except Exception as e:
                    logger.error(f"---ERROR WHILE TERMINATING THREAD: {str(e)}---")
                raise TimeoutError(f"Function call exceeded {timeout_seconds} seconds")
    
    return wrapper


class RAGPipeline:
    """
    Comprehensive RAG Pipeline for Legal Document Processing.
    
    This class orchestrates the entire Retrieval-Augmented Generation workflow
    for legal document processing, implementing a sophisticated state machine
    that handles document retrieval, question routing, answer generation, and
    quality control mechanisms.
    
    The pipeline supports multiple data sources:
    - Local document vector databases
    - Web search for external legal resources
    - Structured legal databases (CanLII)
    
    Key Features:
    - Intelligent question routing based on content analysis
    - Document relevance filtering with AI evaluation
    - Hallucination detection and prevention
    - Multi-source information integration
    - Robust error handling and timeout protection
    
    Attributes:
        llm: Language model instance for text generation and evaluation
        agent_llm: Specialized LLM for agent tasks (routing, grading)
        database_manager: DatabaseManager for document storage and retrieval
        web_search_tool: TavilySearch instance for web-based research
        graph: LangGraph state machine for workflow orchestration
        debug: Debug mode flag for verbose logging
    """

    def __init__(self, llm, database_manager=None, debug=False):
        """
        Initialize the RAG pipeline with language models and data sources.
        
        Sets up the complete RAG workflow including document retrieval,
        question routing, answer generation, and quality control mechanisms.
        
        Args:
            llm: Primary language model instance for text generation
            database_manager (DatabaseManager, optional): Database manager for 
                document storage and retrieval operations
            debug (bool): Enable debug mode for verbose logging
        """
        # Initialize with LLM and optional database manager
        self.agent_llm = llm
        self.database_manager = database_manager
        self.web_search_tool = TavilySearch(k=3)
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def format_docs(self, docs:list[Document]) -> str:
        """
        Format a list of documents into a single string for context.
        
        Args:
            docs: List of Document objects containing content
            
        Returns:
            String containing all document content joined by newlines
        """
        # Join all document contents for context
        formatted_content = "\n\n".join(doc.metadata.get('source_type', 'Document')+':\n'+doc.page_content for doc in docs)
        
        # Check if content is too large (limit to 100KB to prevent LLM issues)
        max_content_size = 100000  # 100KB separate from instructions and prompts
        if len(formatted_content) > max_content_size:
            logger.info(f"---WARNING: Content too large ({len(formatted_content)} chars), truncating to {max_content_size} chars---")
            formatted_content = formatted_content[:max_content_size] + "\n\n[CONTENT TRUNCATED]"
        
        return formatted_content

    def retrieve(self, state):
        """
        Retrieve relevant documents from vector store based on question.
        
        Args:
            state: Current workflow state containing question and client details
            
        Returns:
            Updated state with retrieved documents
        """
        # Retrieve relevant documents from vector store
        logger.info("---RETRIEVE---")
        question = state["question"]
        client_notes = state.get("client_notes", "")
        documents_to_retrieve = state.get("documents_to_retrieve", [])

        # Create enhanced query that includes client details for better retrieval
        enhanced_query = f"{client_notes} Current question: {question}" if client_notes else question
        # TODO: UUids to retrieve if required
        # Query related documents from vector store
        results = self.database_manager.query_documents(enhanced_query, k=10, meta_filter=None) # Top k
        ## Query vector store for specific documents if provided
        expected_docs = self.database_manager.case_vector_store_manager.vector_store.get_by_ids(documents_to_retrieve) if documents_to_retrieve else []
        #logger.info(type(results), type(expected_docs))
        #logger.info(results, expected_docs)
        #documents = set(results + expected_docs)  # Combine results with expected documents
        logger.info(f"{len(results)} documents retrieved from vector store")
        for res in results:
            logger.info(f"------ Retrieved document: {res[0].metadata.get('source_type', 'Unknown')} - {res[0].page_content[:50]}... of length {len(res[0].page_content)} chars, with score: {res[1]}")
        return {"documents": [res[0] for res in results]}

    def generate(self, state):
        """
        Generate answer using retrieved documents
        
        Args:
            state: Current workflow state containing question and conversation history
            
        Returns:
            Updated state with generated answer
        """
        # Generate answer using retrieved documents
        logger.info("---GENERATE---")
        question = state["question"]
        conversation_history = state.get("conversation_history", "First question.")
        case_details = state.get("case_details", "No case details provided.")
        documents = state["documents"]

        content = self.format_docs(documents)

        generation = QuestionReply(self.agent_llm).invoke({
            "question": question,
            "conversation_history": conversation_history,
            "case_details": case_details,
            "context": content
        })
        return {"generation": generation.content, "loop_step": 1}
    
    def grade_documents(self, state):
        """
        Grade each document for relevance to the question.
        
        Args:
            state: Current workflow state containing question and retrieved documents
            
        Returns:
            Updated state with filtered documents and web search flag
        """
        # Grade each document for relevance to the question
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        web_pages = state.get("included_web_pages", [])
        
        # If no documents were retrieved, automatically trigger web search
        if not documents:
            logger.info("---NO DOCUMENTS RETRIEVED: TRIGGERING WEB SEARCH---")
            return {"documents": web_pages, "web_search": "Yes"}
        
        filtered_docs = web_pages
        additional_web_search = "No" 
        web_search_threshold = 2
        for d in documents:
            content = self.format_docs([d])
            score = DocumentGrader(self.agent_llm).invoke({"document": content, "question": question})
            grade = score.binary_score
            logger.info(score.explanation)
            if grade.lower() == "1":
                logger.info(f"---GRADE: DOCUMENT RELEVANT--- : {d.metadata.get('source_type', 'Unknown'), d.page_content[:50]} ...  with score: {score.binary_score}")
                filtered_docs.append(d)
            else:
                logger.info(f"---GRADE: DOCUMENT NOT RELEVANT--- : {d.metadata.get('source_type', 'Unknown'), d.page_content[:50]} ...  with score: {score.binary_score}")
                web_search_threshold -= 1
                if web_search_threshold <= 0:
                    additional_web_search = "Yes"
                    logger.info("---ADDITIONAL WEB SEARCH REQUIRED---")
                    
        return {"documents": filtered_docs, "web_search": additional_web_search}
    
    def no_context_generation(self, state):
        """
        Generate answer without context if no documents found.
        
        Args:
            state: Current workflow state containing question and conversation history
            
        Returns:
            Generated answer
        """
        # Generate answer without context if no documents found
        logger.info("---NO CONTEXT GENERATION---")
        question = state["question"]
        conversation_history = state.get("conversation_history", "First question.")
        case_details = state.get("case_details", "No case details provided.")

        generation = NoContextQuestionReply(self.agent_llm).invoke({
            "question": question,
            "conversation_history": conversation_history,
            "case_details": case_details,
        })
        return {"generation": generation.content, "loop_step": 0}
    
    def web_search(self, state):
        """
        Run web search and add results to documents.
        
        Args:
            state: Current workflow state containing question and existing documents
            
        Returns:
            Updated state with additional documents from web search
        """
        # Run web search and add results to documents
        logger.info("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])
        import json
        with open('jurisdiction_code_names.json', 'r') as f:
            jurisdiction_data = json.load(f)
        
        jurisdiction_name = jurisdiction_data.get(self.database_manager.case.jurisdiction_code, "Unknown jurisdiction")

        # Create enhanced search query with conversation context
        search_query = f"{jurisdiction_name} + {question}"

        results = self.web_search_tool.invoke({"query": search_query})
        results = results.get("results", [])
        logger.info(f"---WEB SEARCH RESULTS--- Found {len(results)} results")
        for d in results:
            if d.get("score", 0) > 0.45:
                documents.append(Document(page_content=d.get("content", ""), metadata={"title": d.get("title", ""), "url": d.get("url", "")}))
        return {"documents": documents}

    def route_question(self, state):
        """
        Decide whether to use web search or vector store based on question.
        
        Args:
            state: Current workflow state containing question and conversation history
            
        Returns:
            Routing decision as a string
        """
        # Decide whether to use web search or vector store
        logger.info("---ROUTE QUESTION---")
        routing_question = state["question"]
        conversation_history = state.get("conversation_history", "")
        
        # Include conversation context in routing decision
        
        source = RagQuestionRouter(self.agent_llm).invoke({"question": routing_question, 'conversation_history': conversation_history or ""})
        #logger.info(source.explanation)

        # TODO: Handle hybrid routing
        if '#' in routing_question or source.datasource == 'DocumentSearch':
            logger.info("---ROUTE QUESTION TO DOCUMENT SEARCH---")
            return "retrieve"

        elif source.datasource == 'WebSearch':
            logger.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        else:
            logger.info("---ROUTE QUESTION TO GENERAL---")
            return "generalreply"

    def decide_to_generate(self, state):
        """
        Decide whether to generate answer or run web search based on document relevance.
        
        Args:
            state: Current workflow state containing web search flag
            
        Returns:
            Decision as a string indicating next action
        """
        # Decide whether to generate answer or run web search
        logger.info("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]

        if web_search == "Yes":
            logger.info("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            logger.info("---DECISION: GENERATE---")
            return "generate"

    def _grade_generation_no_documents(self, state):
        question = state["question"]
        generation = state["generation"]
        
        logger.info("---GRADE GENERATION vs QUESTION (NO DOCUMENTS)---")
        
        score = AnswerGrader(self.agent_llm).invoke({"question": question, "generation": generation})
            
        grade = score.binary_score
        logger.info(f"---DEBUG: Answer grade (no docs): {grade}---")
        logger.info(f"---DEBUG: Answer explanation (no docs): {score.explanation}---")
        
        if grade == "1":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION (NO DOCUMENTS)---")
            return "useful"
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION (NO DOCUMENTS)---")
            return "max retries"



    def grade_generation_v_documents_and_question(self, state):
        """
        Grade the generated answer for grounding and relevance to the question.
        
        Args:
            state: Current workflow state containing question, documents, and generated answer
            
        Returns:
            Decision on the quality of the generation and next action
        """
        # Grade the generated answer for grounding and relevance
        logger.info("---CHECK HALLUCINATIONS---")
        
        # Validate required state components
        if not state.get("question"):
            logger.info("---ERROR: No question found in state---")
            return "max retries"
        if not state.get("generation"):
            logger.info("---ERROR: No generation found in state---")
            return "max retries"
        if not state.get("documents") or len(state["documents"]) == 0:
            logger.info("---ERROR: No documents found in state - skipping hallucination check---")
            # If we have a generation but no documents, we can't check for hallucinations
            # Check if the answer addresses the question
            return self._grade_generation_no_documents(state)
            
            
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        max_retries = state.get("max_retries", 3) # Default to 3 if not provided
        
        content = self.format_docs(documents)
        logger.info(f"---DEBUG: Included document content length: {len(content)} characters---")
        
        logger.info("---DEBUG: Invoking HallucinationGrader---")
                
        score = HallucinationGrader(self.agent_llm).invoke({"documents": content, "generation": generation})
            
        logger.info("---DEBUG: HallucinationGrader completed---")
        
        grade = score.binary_score
        logger.info(f"---DEBUG: Hallucination grade: {grade}---")
        logger.info(f"---DEBUG: Hallucination explanation: {score.explanation}---")

        if grade == "1":
            logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            logger.info("---GRADE GENERATION vs QUESTION---")
            
            logger.info("---DEBUG: Invoking AnswerGrader---")

            score = AnswerGrader(self.agent_llm).invoke({"question": question, "generation": generation})
                
            logger.info("---DEBUG: AnswerGrader completed---")
            
            grade = score.binary_score
            logger.info(f"---DEBUG: Answer grade: {grade}---")
            logger.info(f"---DEBUG: Answer explanation: {score.explanation}---")
            
            if grade == "1":
                logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            elif state["loop_step"] >= max_retries:
                logger.info("---DECISION: MAX RETRIES REACHED---")
                return "max retries"
            else:
                logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        elif state["loop_step"] >= max_retries:
            logger.info("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
        else:
            logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

# Define the state dictionary for the workflow
class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question : str # User question
    generation : str # LLM generation
    web_search : str # Binary decision to run web search
    max_retries : int # Max number of retries for answer generation 
    answers : int # Number of answers generated
    loop_step: Annotated[int, operator.add] 
    documents : List[str] # List of retrieved documents
    conversation_history : str # Previous conversation context
    documents_to_retrieve : List[str] # Specific documents to retrieve by UUIDs
    case_details : str # Case details for context

# Build the workflow graph for the RAG pipeline
def build_workflow(rag_pipeline):
    workflow = StateGraph(GraphState)
    # Add nodes for each step in the pipeline
    workflow.add_node("websearch", rag_pipeline.web_search) # web search
    workflow.add_node("retrieve", rag_pipeline.retrieve) # retrieve
    workflow.add_node("grade_documents", rag_pipeline.grade_documents) # grade documents
    workflow.add_node("generate", rag_pipeline.generate) # generate
    workflow.add_node("generalreply", rag_pipeline.no_context_generation) # general reply without context

    # Set entry point based on question routing
    workflow.set_conditional_entry_point(
        rag_pipeline.route_question,
        {
            "websearch": "websearch",
            "retrieve": "retrieve",
            "generalreply": "generalreply",
        },
    )
    # Define edges between nodes
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("generalreply", END)

    workflow.add_conditional_edges(
        "grade_documents",
        rag_pipeline.decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        rag_pipeline.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )
    return workflow.compile()
