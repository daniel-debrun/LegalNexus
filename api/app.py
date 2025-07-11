"""
Legal RAG Pipeline - Flask REST API

This module provides the REST API interface for the Legal RAG Pipeline application.
It handles all HTTP requests from the frontend and coordinates interactions with
the LegalRagAgent and DatabaseManager components.

The API provides endpoints for:
- Agent initialization and configuration
- Case and client management
- Document upload and search
- Legal question answering
- Chat history management

Author: Legal RAG Pipeline Team
Version: 1.0.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import sys

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.agent import LegalRagAgent
from src.storage.database_manager import DatabaseManager

# Initialize Flask application
app = Flask(__name__)

# Enable CORS for all routes to allow frontend access
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables from .env file
load_dotenv()

# Global instances - initialized when agent is created
agent: LegalRagAgent = None
db_manager: DatabaseManager = None
message_count = 0  # Track number of questions asked for advice generation

# HEALTH CHECK AND STATUS ENDPOINTS

@app.route('/')
def index():
    """
    Root endpoint to verify the API is running.
    
    Returns:
        JSON response with API status
    """
    return jsonify({
        'status': 'ok',
        'message': 'Legal RAG Pipeline API is running',
        'version': '1.0.0',
        'endpoints': {
            'initialize': '/api/initialize_agent',
            'cases': '/api/get_cases',
            'chat': '/api/ask_question'
        }
    })


# AGENT INITIALIZATION AND CONFIGURATION

@app.route('/api/initialize_agent', methods=['POST'])
def initialize_agent():
    """
    Initialize the LegalRagAgent with specified model and directories.
    
    This endpoint sets up the entire legal RAG system including:
    - Language model configuration
    - Database connections
    - Vector store initialization
    
    Request Body:
        model_name (str): Name of the language model to use
        
    Returns:
        JSON response indicating success/failure of initialization
    """
    data = request.json
    model_name = data.get('model_name', 'mistral-large-2411')
    
    global agent, db_manager
    
    # If agent already exists, just update the model
    if agent:
        if agent.set_llm(model_name):
            return jsonify({
                'success': True,
                'message': f'Agent re-initialized with model {model_name}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'message': f'Model {model_name} not supported'
            }), 500
        
    try:
        # Validate required environment variables
        db_dir = os.getenv('DB_DIR')
        vector_store_dir = os.getenv('VECTOR_STORE_DIR')

        if not db_dir or not vector_store_dir:
            raise EnvironmentError(
                "Required environment variables DB_DIR and VECTOR_STORE_DIR are not set. "
                "Please check your .env file."
            )

        # Create necessary directories if they don't exist
        for dir_path in [db_dir, vector_store_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        # Initialize database manager
        db_manager = DatabaseManager(
            root_vector_db_dir=vector_store_dir,
            db_dir=db_dir,
        )
        
        # Initialize the legal RAG agent
        agent = LegalRagAgent(db_manager, model_name=model_name)
        
        return jsonify({
            'success': True,
            'message': f'Legal RAG Agent initialized successfully with {model_name}',
            'model': model_name
        }), 200
        
    except EnvironmentError as e:
        return jsonify({
            'success': False,
            'message': f'Configuration error: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error initializing agent: {str(e)}'
        }), 500


# CASE MANAGEMENT ENDPOINTS

@app.route('/api/get_cases', methods=['GET'])
def get_cases():
    """
    Retrieve all cases, optionally filtered by search term.
    
    Query Parameters:
        search_term (str, optional): Filter cases by name containing this term
        
    Returns:
        JSON response with list of cases matching the search criteria
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    search_term = request.args.get('search_term', '')
    
    try:
        cases = agent.database_manager.get_cases_by_name(search_term)
        case_list = []
        
        for case in cases:
            case_list.append({
                'id': case.id,
                'name': case.name,
                'jurisdiction_code': case.jurisdiction_code,
                'notes': case.notes,
                'created_at': case.created_at.isoformat() if case.created_at else None,
                'case_type': case.case_type,
                'case_status': case.case_status,
                'client_id': case.client_id
            })
            
        return jsonify({
            'success': True, 
            'cases': case_list,
            'total': len(case_list)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error fetching cases: {str(e)}'
        }), 500

@app.route('/api/load_case', methods=['POST'])
def load_case():
    """
    Load a specific case for the current session.
    
    This endpoint loads a case and initializes the database manager with
    the case context, enabling case-specific operations like document
    management and chat history.
    
    Request Body:
        case_name (str): Name of the case to load
        case_path (str, optional): Path to JSON file containing case data
        
    Returns:
        JSON response with loaded case information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    data = request.json
    case_path = data.get('case_path', '')
    case_name = data.get('case_name', '')
    
    try:
        # Load case from JSON file or by name
        if case_path:
            case = agent._new_object_from_json("case", case_path)
        else:
            case = agent.get_case_by_name_web_safe(case_name)
        
        if not case:
            return jsonify({
                'success': False, 
                'message': f'Case "{case_name}" not found. Please check the case name or create a new case.'
            }), 404
        
        # Initialize case context in database manager
        agent.database_manager.initialize_from_case(case)
        agent.rag_pipeline.database_manager = agent.database_manager
        
        return jsonify({
            'success': True,
            'message': f'Case "{case.name}" loaded successfully',
            'case': {
                'id': case.id,
                'name': case.name,
                'jurisdiction_code': case.jurisdiction_code,
                'notes': case.notes,
                'created_at': case.created_at.isoformat() if case.created_at else None,
                'case_type': case.case_type,
                'case_status': case.case_status,
                'client_id': case.client_id
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error loading case: {str(e)}'
        }), 500

@app.route('/api/create_case', methods=['POST'])
def create_case():
    """
    Create a new legal case.
    
    Request Body:
        name (str): Case name
        jurisdiction_code (str): Legal jurisdiction code (e.g., 'oncj', 'bcca')
        notes (str, optional): Additional case notes
        case_type (str, optional): Type of case (e.g., 'Civil', 'Criminal')
        case_status (str, optional): Current status (e.g., 'Open', 'Closed')
        
    Returns:
        JSON response with created case information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    data = request.json
    
    # Validate required fields
    if not data.get('name'):
        return jsonify({
            'success': False, 
            'message': 'Case name is required'
        }), 400
    
    try:
        case = agent.database_manager.add_case(**data)
        
        # Initialize the case context
        agent.database_manager.initialize_from_case(case)
        agent.rag_pipeline.database_manager = agent.database_manager
        
        return jsonify({
            'success': True,
            'message': f'Case "{case.name}" created successfully',
            'case': {
                'id': case.id,
                'name': case.name,
                'jurisdiction_code': case.jurisdiction_code,
                'notes': case.notes,
                'case_type': getattr(case, 'case_type', None),
                'case_status': getattr(case, 'case_status', None)
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error creating case: {str(e)}'
        }), 500

@app.route('/api/update_case', methods=['PUT'])
def update_case():
    """
    Update an existing case.
    
    Request Body:
        case_id (str): ID of the case to update
        name (str, optional): Case name
        jurisdiction_code (str, optional): Legal jurisdiction code
        notes (str, optional): Additional case notes
        case_type (str, optional): Type of case
        case_status (str, optional): Current status
        
    Returns:
        JSON response with updated case information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    data = request.json
    case_id = data.get('case_id')
    
    if not case_id:
        return jsonify({
            'success': False, 
            'message': 'Case ID is required for updates'
        }), 400
    
    try:
        # Get the existing case
        case = agent.database_manager.get_case_by_id(case_id)
        if not case:
            return jsonify({
                'success': False, 
                'message': f'Case with ID "{case_id}" not found'
            }), 404
        
        # Update only the provided fields
        if 'name' in data:
            case.name = data['name']
        if 'jurisdiction_code' in data:
            case.jurisdiction_code = data['jurisdiction_code']
        if 'notes' in data:
            case.notes = data['notes']
        if 'case_type' in data:
            case.case_type = data['case_type']
        if 'case_status' in data:
            case.case_status = data['case_status']
        
        # Commit changes and refresh
        agent.database_manager.session.commit()
        agent.database_manager.session.refresh(case)
        
        # If this is the current case, update the agent's reference
        if agent.database_manager.case and agent.database_manager.case.id == case.id:
            agent.database_manager.case = case
        
        return jsonify({
            'success': True,
            'message': f'Case "{case.name}" updated successfully',
            'case': {
                'id': case.id,
                'name': case.name,
                'jurisdiction_code': case.jurisdiction_code,
                'notes': case.notes,
                'case_type': getattr(case, 'case_type', None),
                'case_status': getattr(case, 'case_status', None),
                'client_id': case.client_id
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error updating case: {str(e)}'
        }), 500


# CLIENT MANAGEMENT ENDPOINTS

@app.route('/api/get_clients', methods=['GET'])
def get_clients():
    """
    Retrieve all clients, optionally filtered by search term, or a specific client by ID.
    
    Query Parameters:
        client_id (str, optional): ID of specific client to retrieve
        search_term (str, optional): Filter clients by name containing this term
        
    Returns:
        JSON response with client details or list of clients
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    client_id = request.args.get('client_id', '')
    search_term = request.args.get('search_term', '')
    
    try:
        # If client_id is provided, return specific client (backward compatibility)
        if client_id:
            client = agent.database_manager.get_client_by_id(client_id)
            
            if client:
                client_data = {
                    'id': client.id,
                    'name': client.name,
                    'email': client.email,
                    'phone': client.phone,
                    'address': client.address,
                    'date_of_birth': client.date_of_birth,
                    'gender': client.gender,
                    'occupation': client.occupation,
                    'notes': client.notes
                }
                return jsonify({
                    'success': True, 
                    'client': client_data
                }), 200
            else:
                return jsonify({
                    'success': False, 
                    'message': f'Client with ID "{client_id}" not found'
                }), 404
        
        # Otherwise, return all clients filtered by search term
        else:
            clients = agent.database_manager.get_clients_by_name(search_term)
            client_list = []
            
            for client in clients:
                client_list.append({
                    'id': client.id,
                    'name': client.name,
                    'email': client.email,
                    'phone': client.phone,
                    'address': client.address,
                    'date_of_birth': client.date_of_birth,
                    'gender': client.gender,
                    'occupation': client.occupation,
                    'notes': client.notes
                })
                
            return jsonify({
                'success': True, 
                'clients': client_list
            }), 200
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error fetching clients: {str(e)}'
        }), 500

@app.route('/api/load_client', methods=['POST'])
def load_client():
    """
    Load a client for the current case session.
    
    Request Body:
        client_name (str): Name of the client to load
        
    Returns:
        JSON response with loaded client information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded before loading a client'
        }), 400
    
    data = request.json
    client_name = data.get('client_name', '')
    
    if not client_name:
        return jsonify({
            'success': False, 
            'message': 'Client name is required'
        }), 400
    
    try:
        client = agent.get_client_by_name_web_safe(client_name)
        
        if not client:
            return jsonify({
                'success': False, 
                'message': f'Client "{client_name}" not found. Please create the client first.'
            }), 404
        
        return jsonify({
            'success': True,
            'message': f'Client "{client.name}" loaded successfully',
            'client': {
                'id': client.id,
                'name': client.name,
                'email': client.email,
                'phone': client.phone,
                'address': client.address,
                'date_of_birth': client.date_of_birth,
                'gender': client.gender,
                'occupation': client.occupation,
                'notes': client.notes
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error loading client: {str(e)}'
        }), 500

@app.route('/api/create_client', methods=['POST'])
def create_client():
    """
    Create a new client and associate with the current case.
    
    Request Body:
        name (str): Client's full name
        client_details (dict): Personal information including:
            - email (str): Email address
            - phone (str): Phone number
            - address (str): Physical address
            - date_of_birth (str): Date of birth
            - gender (str): Gender identity
            - occupation (str): Occupation
        notes (str, optional): Additional notes about the client
        
    Returns:
        JSON response with created client information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded before creating a client'
        }), 400
    
    data = request.json
    
    # Validate required fields
    if not data.get('name'):
        return jsonify({
            'success': False, 
            'message': 'Client name is required'
        }), 400
    
    try:
        client = agent.database_manager.add_case_client(
            case=agent.database_manager.case, 
            **data
        )
        
        return jsonify({
            'success': True,
            'message': f'Client "{client.name}" created successfully',
            'client': {
                'id': client.id,
                'name': client.name,
                'email': client.email,
                'phone': client.phone,
                'address': client.address,
                'date_of_birth': client.date_of_birth,
                'gender': client.gender,
                'occupation': client.occupation
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error creating client: {str(e)}'
        }), 500

@app.route('/api/update_client', methods=['PUT'])
def update_client():
    """
    Update an existing client.
    
    Request Body:
        client_id (str): ID of the client to update
        name (str, optional): Client's full name
        email (str, optional): Email address
        phone (str, optional): Phone number
        address (str, optional): Physical address
        date_of_birth (str, optional): Date of birth
        gender (str, optional): Gender identity
        occupation (str, optional): Occupation
        notes (str, optional): Additional notes about the client
        
    Returns:
        JSON response with updated client information
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    data = request.json
    client_id = data.get('client_id')
    
    if not client_id:
        return jsonify({
            'success': False, 
            'message': 'Client ID is required for updates'
        }), 400
    
    try:
        # Get the existing client
        client = agent.database_manager.get_client_by_id(client_id)
        if not client:
            return jsonify({
                'success': False, 
                'message': f'Client with ID "{client_id}" not found'
            }), 404
        
        # Update only the provided fields
        if 'name' in data:
            client.name = data['name']
        if 'notes' in data:
            client.notes = data['notes']
        
        # Handle client_details structure - update the JSON field properly
        client_details = client.client_details.copy() if client.client_details else {}
        
        # Update client_details from both flat structure and nested structure
        if 'email' in data:
            client_details['email'] = data['email']
        if 'phone' in data:
            client_details['phone'] = data['phone']
        if 'address' in data:
            client_details['address'] = data['address']
        if 'date_of_birth' in data:
            client_details['date_of_birth'] = data['date_of_birth']
        if 'gender' in data:
            client_details['gender'] = data['gender']
        if 'occupation' in data:
            client_details['occupation'] = data['occupation']
            
        # Also handle nested client_details structure for backward compatibility
        nested_details = data.get('client_details', {})
        if 'email' in nested_details:
            client_details['email'] = nested_details['email']
        if 'phone' in nested_details:
            client_details['phone'] = nested_details['phone']
        if 'address' in nested_details:
            client_details['address'] = nested_details['address']
        if 'date_of_birth' in nested_details:
            client_details['date_of_birth'] = nested_details['date_of_birth']
        if 'gender' in nested_details:
            client_details['gender'] = nested_details['gender']
        if 'occupation' in nested_details:
            client_details['occupation'] = nested_details['occupation']
        
        # Update the client_details JSON field
        client.client_details = client_details
        
        # Commit changes and refresh
        agent.database_manager.session.commit()
        agent.database_manager.session.refresh(client)
        
        # If this is the current client, update the agent's reference
        if agent.database_manager.client and agent.database_manager.client.id == client.id:
            agent.database_manager.client = client
        
        return jsonify({
            'success': True,
            'message': f'Client "{client.name}" updated successfully',
            'client': {
                'id': client.id,
                'name': client.name,
                'email': client.email,
                'phone': client.phone,
                'address': client.address,
                'date_of_birth': client.date_of_birth,
                'gender': client.gender,
                'occupation': client.occupation,
                'notes': client.notes
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error updating client: {str(e)}'
        }), 500


# LEGAL REFERENCE AND RESEARCH ENDPOINTS

@app.route('/api/set_legal_references', methods=['POST'])
def set_legal_references():
    """
    Set up legal references for the current case by searching CanLII.
    
    This endpoint automatically searches Canadian legal databases for
    relevant case law and legislation based on the current case details.
    The AI evaluates and filters the most relevant legal precedents.
    
    Returns:
        JSON response indicating success/failure of reference setup
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded before setting legal references'
        }), 400
    
    try:
        # Search CanLII and set up legal references
        agent.set_legal_references()
        
        return jsonify({
            'success': True, 
            'message': 'Legal references have been successfully researched and set for this case'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error setting legal references: {str(e)}'
        }), 500


# CHAT AND QUESTION ANSWERING ENDPOINTS

@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    """
    Process a legal question and provide an AI-generated response.
    
    This endpoint is the core of the legal assistance functionality.
    It uses the RAG pipeline to provide contextual answers based on:
    - Uploaded case documents
    - Legal precedents and references
    - Case and client context
    - Conversation history
    
    Request Body:
        question (str): The legal question to ask
        
    Returns:
        JSON response with the AI-generated answer and metadata
    """
    global agent, message_count
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded before asking questions'
        }), 400
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({
            'success': False, 
            'message': 'Question cannot be empty'
        }), 400
    
    try:
        # Process the question through the RAG pipeline
        chat_interaction = agent.execute_actions(question)
        
        if not chat_interaction:
            return jsonify({
                'success': False, 
                'message': 'Failed to process the question. Please try again.'
            }), 500
        
        # Increment message count for advice generation tracking
        message_count += 1
        
        response_data = {
            'success': True,
            'question': chat_interaction.user_prompt,
            'answer': chat_interaction.assistant_response,
            'timestamp': chat_interaction.timestamp.isoformat() if chat_interaction.timestamp else None,
            'message_count': message_count,
        }
        
        # Generate periodic legal advice (every 5 messages)
        if message_count % 5 == 0:
            try:
                advice_response = agent.generate_passive_legal_information()
                if advice_response:
                    response_data['new_advice'] = advice_response
            except Exception as advice_error:
                # Don't fail the whole request if advice generation fails
                print(f"Warning: Failed to generate passive legal advice: {advice_error}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error processing question: {str(e)}'
        }), 500

@app.route('/api/get_chat_history', methods=['GET'])
def get_chat_history():
    """
    Retrieve the chat history for the current case.
    
    Returns:
        JSON response with list of previous questions and answers
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded to retrieve chat history'
        }), 400
    
    try:
        chat_history = agent.database_manager.get_chat_history()
        history_list = []
        
        for chat in chat_history:
            history_list.append({
                'id': chat.id,
                'question': chat.user_prompt,
                'answer': chat.assistant_response,
                'timestamp': chat.timestamp.isoformat() if chat.timestamp else None
            })
            
        return jsonify({
            'success': True, 
            'history': history_list,
            'total': len(history_list)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error fetching chat history: {str(e)}'
        }), 500


# DOCUMENT MANAGEMENT ENDPOINTS

@app.route('/api/get_documents', methods=['GET'])
def get_documents():
    """
    Retrieve all documents uploaded for the current case.
    
    Returns:
        JSON response with list of uploaded documents and their metadata
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded to retrieve documents'
        }), 400
    
    try:
        documents = agent.database_manager.get_uploaded_documents()
        doc_list = []
        
        for doc in documents:
            doc_list.append({
                'id': doc.id,
                'source_name': doc.source_name,
                'title': doc.title,
                'description': doc.description,
                'language': doc.language,
                'uploaded_at': doc.uploaded_at.isoformat() if doc.uploaded_at else None
            })
            
        return jsonify({
            'success': True, 
            'documents': doc_list,
            'total': len(doc_list)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error fetching documents: {str(e)}'
        }), 500

@app.route('/api/upload_document', methods=['POST'])
def upload_document():
    """
    Upload documents to the current case for semantic search and analysis.
    
    This endpoint accepts both local file paths and web URLs, processes them
    through OCR if needed, and indexes them in the vector database for
    intelligent retrieval during legal consultations.
    
    Request Body:
        file_paths (list): List of local file paths to upload
        urls (list): List of web URLs to scrape and index
        
    Returns:
        JSON response indicating upload success and number of documents processed
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded before uploading documents'
        }), 400
    
    data = request.json
    file_paths = data.get('file_paths', [])
    urls = data.get('urls', [])
    
    if not file_paths and not urls:
        return jsonify({
            'success': False, 
            'message': 'Either file paths or URLs must be provided'
        }), 400
    
    try:
        # Upload and index documents
        result = agent.upload_documents(web_urls=urls, doc_files=file_paths)
        
        total_uploaded = len(file_paths) + len(urls)
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded and indexed {total_uploaded} documents',
            'files_processed': len(file_paths),
            'urls_processed': len(urls)
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error uploading documents: {str(e)}'
        }), 500

@app.route('/api/delete_document', methods=['DELETE'])
def delete_document():
    """
    Delete a document from the current case.
    
    This removes the document from both the database and vector store,
    ensuring it won't appear in future search results.
    
    Request Body:
        document_id (str): ID of the document to delete
        
    Returns:
        JSON response indicating deletion success/failure
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    if not agent.database_manager.case:
        return jsonify({
            'success': False, 
            'message': 'A case must be loaded to delete documents'
        }), 400
    
    data = request.json
    document_id = data.get('document_id', '')
    
    if not document_id:
        return jsonify({
            'success': False, 
            'message': 'Document ID is required'
        }), 400
    
    try:
        success = agent.database_manager.delete_documents([document_id])
        
        if not success:
            return jsonify({
                'success': False, 
                'message': f'Document with ID "{document_id}" not found'
            }), 404
        
        return jsonify({
            'success': True, 
            'message': 'Document deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error deleting document: {str(e)}'
        }), 500


# UTILITY AND SYSTEM ENDPOINTS

@app.route('/api/save', methods=['POST'])
def save():
    """
    Save all pending changes to the database and vector store.
    
    This endpoint commits all database transactions and saves the
    current state of the vector store to disk.
    
    Returns:
        JSON response indicating save success/failure
    """
    global agent
    if not agent:
        return jsonify({
            'success': False, 
            'message': 'Agent not initialized. Please initialize the agent first.'
        }), 400
    
    try:
        success = agent.database_manager.save_all()
        
        if success:
            return jsonify({
                'success': True, 
                'message': 'All changes saved successfully'
            }), 200
        else:
            return jsonify({
                'success': True, 
                'message': 'No changes to save'
            }), 200
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error saving changes: {str(e)}'
        }), 500

@app.route('/api/get_general_advice', methods=['GET'])
def get_general_advice():
    """
    Generate general legal advice and helpful resources.
    
    This endpoint provides periodic legal guidance and educational
    resources that may be relevant to legal practice in general.
    
    Returns:
        JSON response with general legal advice and resource links
    """
    global agent
    
    try:
        if agent and agent.database_manager.case:
            # Generate contextual advice if case is loaded
            advice_response = agent.generate_passive_legal_information()
            passive_advice = advice_response.get('advice', 'No specific advice available at this time.')
            resources = advice_response.get('helpful_resources', [])
        else:
            # Provide generic legal guidance
            passive_advice = (
                "Legal processes often involve several key steps: identifying the appropriate "
                "forms and procedures, completing and submitting required documents, understanding "
                "deadlines and timelines, and following ethical guidelines for honesty and "
                "confidentiality. Whether you are filing a case, responding to legal notices, "
                "or simply completing legal forms, it is important to ensure accuracy and "
                "completeness. Procedures and requirements can vary significantly by jurisdiction "
                "and case type, so consulting with qualified legal professionals is always recommended "
                "for specific legal matters."
            )
            resources = [
                "Government of Canada Justice Laws Website: https://laws-lois.justice.gc.ca/",
                "CanLII - Canadian Legal Information Institute: https://www.canlii.org/",
                "Legal Aid Offices (varies by province)"
            ]
        
        return jsonify({
            'success': True,
            'advice': passive_advice,
            'resources': resources,
            'disclaimer': 'This information is for educational purposes only and does not constitute legal advice.'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating general advice: {str(e)}'
        }), 500


# APPLICATION STARTUP

if __name__ == '__main__':
    """
    Start the Flask development server.
    
    For production deployment, use a proper WSGI server like Gunicorn.
    """
    print("Starting Legal RAG Pipeline API Server...")
    print("API will be available at: http://localhost:5000")
    
    app.run(
        debug=True,      # Enable debug mode for development
        host='0.0.0.0',  # Listen on all interfaces
        port=5000        # Default port
    )