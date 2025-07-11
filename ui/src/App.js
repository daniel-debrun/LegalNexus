import React, { useState, useEffect } from 'react';
import { Box, Container, Grid, Paper, ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import Header from './components/Header';
import ChatPanel from './components/ChatPanel';
import LeftPanel from './components/LeftPanel';
import RightPanel from './components/RightPanel';
import SetupDialog from './components/SetupDialog';
import Api from './services/Api';

// Create a professional legal theme
const legalTheme = createTheme({
  palette: {
    primary: {
      main: '#1e3a8a',
      light: '#3b82f6',
      dark: '#1e40af',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#d97706',
      light: '#f59e0b',
      dark: '#b45309',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#1f2937',
      secondary: '#6b7280',
    },
    success: {
      main: '#059669',
    },
    error: {
      main: '#dc2626',
    },
  },
  typography: {
    fontFamily: '"Inter", "Segoe UI", "Roboto", "Helvetica Neue", -apple-system, BlinkMacSystemFont, sans-serif',
    h1: {
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    h2: {
      fontWeight: 700,
      letterSpacing: '-0.025em',
    },
    h3: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    h4: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.025em',
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          '&:hover': {
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.15)',
            transform: 'translateY(-2px)',
          },
          transition: 'all 0.2s ease',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 20px',
          transition: 'all 0.2s ease',
          '&:hover': {
            transform: 'translateY(-1px)',
          },
        },
        contained: {
          boxShadow: '0 4px 12px rgba(30, 58, 138, 0.2)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(30, 58, 138, 0.3)',
          },
        },
      },
    },
    MuiAccordion: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          borderRadius: '12px !important',
          marginBottom: 16,
          '&:before': {
            display: 'none',
          },
        },
      },
    },
    MuiAccordionSummary: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
          borderRadius: '12px 12px 0 0',
          fontWeight: 600,
          '&.Mui-expanded': {
            borderRadius: '12px 12px 0 0',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 8,
            '& fieldset': {
              borderWidth: 2,
              borderColor: '#e5e7eb',
            },
            '&:hover fieldset': {
              borderColor: '#1e3a8a',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#1e3a8a',
            },
          },
        },
      },
    },
  },
});

function App() {
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [openSetup, setOpenSetup] = useState(true);
  const [messages, setMessages] = useState([]);
  const [currentCase, setCurrentCase] = useState(null);
  const [currentClient, setCurrentClient] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState(null);
  const [leftPanelVisible, setLeftPanelVisible] = useState(true);

  // State for message count and legal advice
  const [messageCount, setMessageCount] = useState(parseInt(localStorage.getItem('message_count') || '0', 10));
  const [legalAdvice, setLegalAdvice] = useState(null);

  // Initialize the agent when the component mounts
  useEffect(() => {
    // When component mounts, check if we have case/client info in sessionStorage (clears on tab close)
    // Change to localStorage if you want persistence across browser sessions
    const savedCase = sessionStorage.getItem('currentCase');
    const savedClient = sessionStorage.getItem('currentClient');
    
    if (savedCase) {
      setCurrentCase(JSON.parse(savedCase));
    }
    
    if (savedClient) {
      setCurrentClient(JSON.parse(savedClient));
    }
  }, []);

  // Load chat history when a case is selected
  useEffect(() => {
    if (currentCase && isInitialized) {
      loadChatHistory();
      loadDocuments();
    }
  }, [currentCase, isInitialized]);

  const handleInitialize = async (modelName) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.initialize(modelName);
      if (response.success) {
        setIsInitialized(true);
        setOpenSetup(false);
      } else {
        setError(response.message || 'Failed to initialize agent');
      }
    } catch (err) {
      setError(err.message || 'An error occurred during initialization');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadCase = async (caseName) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.loadCase(caseName);
      if (response.success) {
        setCurrentCase(response.case);
        sessionStorage.setItem('currentCase', JSON.stringify(response.case));
      } else {
        setError(response.message || 'Failed to load case');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while loading the case');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateCase = async (caseData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      let response;
      
      // Check if this is an update (caseData has case_id) or create (no case_id)
      if (caseData.case_id) {
        // Update existing case
        response = await Api.updateCase(caseData);
      } else {
        // Create new case
        response = await Api.createCase(caseData);
      }
      
      if (response.success) {
        setCurrentCase(response.case);
        sessionStorage.setItem('currentCase', JSON.stringify(response.case));
      } else {
        setError(response.message || 'Failed to save case');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while saving the case');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadClient = async (clientName) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.loadClient(clientName);
      if (response.success) {
        setCurrentClient(response.client);
        sessionStorage.setItem('currentClient', JSON.stringify(response.client));
      } else {
        setError(response.message || 'Failed to load client');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while loading the client');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateClient = async (clientData) => {
    setIsLoading(true);
    setError(null);
    
    try {
      let response;
      
      // Check if this is an update (clientData has client_id) or create (no client_id)
      if (clientData.client_id) {
        // Update existing client
        response = await Api.updateClient(clientData);
      } else {
        // Create new client
        response = await Api.createClient(clientData);
      }
      
      if (response.success) {
        setCurrentClient(response.client);
        sessionStorage.setItem('currentClient', JSON.stringify(response.client));
      } else {
        setError(response.message || 'Failed to save client');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while saving the client');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSetLegalReferences = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.setLegalReferences();
      if (!response.success) {
        setError(response.message || 'Failed to set legal references');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while setting legal references');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;
    
    // Add user message to the list
    const userMessage = {
      id: Date.now().toString(),
      content: message,
      role: 'user',
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.askQuestion(message);
      
      if (response.success) {
        // Add assistant message to the list
        const assistantMessage = {
          id: (Date.now() + 1).toString(),
          content: response.answer,
          role: 'assistant',
          timestamp: response.timestamp || new Date().toISOString()
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        
        // Update message count and check for new legal advice
        if (response.message_count) {
          setMessageCount(response.message_count);
        }
        
        if (response.new_advice) {
          setLegalAdvice({
            title: 'Updated Legal Guidance',
            content: response.new_advice.advice,
            resources: response.new_advice.links,
            titles: response.new_advice.link_titles
          });
        }
      } else {
        setError(response.message || 'Failed to get a response');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while processing your message');
    } finally {
      setIsLoading(false);
    }
  };

  const loadChatHistory = async () => {
    setIsLoading(true);
    
    try {
      const response = await Api.getChatHistory();
      if (response.success) {
        const formattedMessages = response.history.flatMap(item => [
          {
            id: `q-${item.id}`,
            content: item.question,
            role: 'user',
            timestamp: item.timestamp
          },
          {
            id: `a-${item.id}`,
            content: item.answer,
            role: 'assistant',
            timestamp: item.timestamp
          }
        ]);
        
        setMessages(formattedMessages);
      }
    } catch (err) {
      setError(err.message || 'Failed to load chat history');
    } finally {
      setIsLoading(false);
    }
  };

  const loadDocuments = async () => {
    setIsLoading(true);
    
    try {
      const response = await Api.getDocuments();
      if (response.success) {
        setDocuments(response.documents);
      }
    } catch (err) {
      setError(err.message || 'Failed to load documents');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadDocument = async (filePaths, urls) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.uploadDocument(filePaths, urls);
      if (response.success) {
        // Refresh documents list
        loadDocuments();
      } else {
        setError(response.message || 'Failed to upload document');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while uploading the document');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteDocument = async (documentId) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.deleteDocument(documentId);
      if (response.success) {
        // Refresh documents list
        loadDocuments();
      } else {
        setError(response.message || 'Failed to delete document');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while deleting the document');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveAll = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await Api.saveAll();
      if (!response.success) {
        setError(response.message || 'Failed to save changes');
      }
    } catch (err) {
      setError(err.message || 'An error occurred while saving changes');
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleLeftPanel = () => {
    setLeftPanelVisible(!leftPanelVisible);
  };

  const handleClearSession = () => {
    // Clear sessionStorage
    sessionStorage.removeItem('currentCase');
    sessionStorage.removeItem('currentClient');
    localStorage.removeItem('message_count'); // This can stay in localStorage as it's a preference
    
    // Reset state
    setCurrentCase(null);
    setCurrentClient(null);
    setMessages([]);
    setDocuments([]);
    setMessageCount(0);
    setLegalAdvice(null);
    setError(null);
  };

  return (
    <ThemeProvider theme={legalTheme}>
      <CssBaseline />
      <Box sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100vh',
        background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
        minHeight: '100vh'
      }}>
        <Header 
          isInitialized={isInitialized} 
          currentCase={currentCase}
          currentClient={currentClient}
          onOpenSetup={() => setOpenSetup(true)}
          onSaveAll={handleSaveAll}
          onToggleLeftPanel={handleToggleLeftPanel}
          onClearSession={handleClearSession}
          leftPanelVisible={leftPanelVisible}
        />
        
        <Box component="main" sx={{ flexGrow: 1, pt: 8 }}>
          <Container maxWidth="xl" sx={{ mt: 2 }}>
            <Grid container spacing={3}>
              {leftPanelVisible && (
                <Grid item xs={12} md={3}>
                  <Paper className="panel" elevation={0} sx={{
                    background: 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(229, 231, 235, 0.8)',
                    borderRadius: 3,
                  }}>
                    <LeftPanel 
                      currentCase={currentCase}
                      currentClient={currentClient}
                      documents={documents}
                      onLoadCase={handleLoadCase}
                      onCreateCase={handleCreateCase}
                      onLoadClient={handleLoadClient}
                      onCreateClient={handleCreateClient}
                      onUploadDocument={handleUploadDocument}
                      onDeleteDocument={handleDeleteDocument}
                      isLoading={isLoading}
                    />
                  </Paper>
                </Grid>
              )}
              
              <Grid item xs={12} md={leftPanelVisible ? 6 : 9}>
                <Paper elevation={0} sx={{ 
                  height: 'calc(100vh - 100px)',
                  background: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(229, 231, 235, 0.8)',
                  borderRadius: 3,
                  overflow: 'hidden'
                }}>
                  <ChatPanel 
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    isLoading={isLoading}
                    error={error}
                    onClearError={() => setError(null)}
                  />
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Paper className="panel" elevation={0} sx={{
                  background: 'rgba(255, 255, 255, 0.95)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(229, 231, 235, 0.8)',
                  borderRadius: 3,
                }}>
                  <RightPanel 
                    currentCase={currentCase}
                    onSetLegalReferences={handleSetLegalReferences}
                    legalAdvice={legalAdvice}
                    messageCount={messageCount}
                  />
                </Paper>
              </Grid>
            </Grid>
          </Container>
        </Box>
        
        <SetupDialog 
          open={openSetup} 
          onClose={() => isInitialized && setOpenSetup(false)}
          onInitialize={handleInitialize}
          isLoading={isLoading}
          error={error}
          onClearError={() => setError(null)}
        />
      </Box>
    </ThemeProvider>
  );
}

export default App;
