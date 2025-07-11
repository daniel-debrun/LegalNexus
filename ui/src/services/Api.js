import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const Api = {
  // Initialize agent
  initialize: async (modelName) => {
    try {
      // Function call (see app.py)
      const response = await axios.post(`${API_URL}/initialize_agent`, {
        model_name: modelName,
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to initialize agent');
    }
  },

  // Case operations
  getCases: async (searchTerm = '') => {
    try {
      const response = await axios.get(`${API_URL}/get_cases`, {
        params: { search_term: searchTerm }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to fetch cases');
    }
  },

  loadCase: async (caseName = '') => {
    try {
      const response = await axios.post(`${API_URL}/load_case`, {
        case_name: caseName
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to load case');
    }
  },

  createCase: async (caseData) => {
    try {
      const response = await axios.post(`${API_URL}/create_case`, caseData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to create case');
    }
  },

  updateCase: async (caseData) => {
    try {
      const response = await axios.put(`${API_URL}/update_case`, caseData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to update case');
    }
  },

  // Client operations
  getClients: async (searchTerm = '') => {
    try {
      const response = await axios.get(`${API_URL}/get_clients`, {
        params: { search_term: searchTerm }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to fetch clients');
    }
  },

  getClient: async (clientId) => {
    try {
      const response = await axios.get(`${API_URL}/get_clients`, {
        params: { client_id: clientId }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to fetch client');
    }
  },

  loadClient: async (clientName = '') => {
    try {
      const response = await axios.post(`${API_URL}/load_client`, {
        client_name: clientName
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to load client');
    }
  },

  createClient: async (clientData) => {
    try {
      const response = await axios.post(`${API_URL}/create_client`, clientData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to create client');
    }
  },

  updateClient: async (clientData) => {
    try {
      const response = await axios.put(`${API_URL}/update_client`, clientData);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to update client');
    }
  },

  // Legal references
  setLegalReferences: async () => {
    try {
      const response = await axios.post(`${API_URL}/set_legal_references`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to set legal references');
    }
  },

  // Chat operations
  askQuestion: async (question) => {
    try {
      const response = await axios.post(`${API_URL}/ask_question`, { question });
      
      // Store the message count in localStorage to persist between sessions
      if (response.data.message_count) {
        localStorage.setItem('message_count', response.data.message_count);
      }
      
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get response');
    }
  },

  getChatHistory: async () => {
    try {
      const response = await axios.get(`${API_URL}/get_chat_history`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to fetch chat history');
    }
  },

  // Document operations
  getDocuments: async () => {
    try {
      const response = await axios.get(`${API_URL}/get_documents`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to fetch documents');
    }
  },

  uploadDocument: async (filePaths = [], urls = []) => {
    try {
      const response = await axios.post(`${API_URL}/upload_document`, { file_paths: filePaths, urls });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to upload document');
    }
  },

  deleteDocument: async (documentId) => {
    try {
      const response = await axios.delete(`${API_URL}/delete_document`, {
        data: { document_id: documentId }
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to delete document');
    }
  },

  // Save all changes
  saveAll: async () => {
    try {
      const response = await axios.post(`${API_URL}/save`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to save changes');
    }
  },
  
  // Get general advice
  getGeneralAdvice: async () => {
    try {
      const response = await axios.get(`${API_URL}/get_general_advice`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get general advice');
    }
  }
};

export default Api;
