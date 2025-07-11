import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  CircularProgress, 
  Alert, 
  Snackbar,
  Avatar,
  Divider
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import ReactMarkdown from 'react-markdown';

const ChatPanel = ({ messages, onSendMessage, isLoading, error, onClearError }) => {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    
    onSendMessage(input);
    setInput('');
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (err) {
      return '';
    }
  };

  return (
    <Box className="chat-container">
      <Box className="message-list">
        {messages.length === 0 && (
          <Box sx={{ 
            textAlign: 'center', 
            py: 8,
            px: 4,
            background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
            borderRadius: 3,
            border: '2px dashed',
            borderColor: 'divider',
            mx: 2
          }}>
            <Avatar sx={{ 
              mx: 'auto',
              mb: 3,
              width: 64,
              height: 64,
              background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)'
            }}>
              <SmartToyIcon sx={{ fontSize: 32 }} />
            </Avatar>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600, color: 'text.primary' }}>
              Legal AI Assistant Ready
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 400, mx: 'auto' }}>
              Ask questions about your case, upload documents for analysis, or request legal research assistance.
            </Typography>
          </Box>
        )}
        
        {messages.map((message, index) => (
          <Box key={message.id}>
            <Box
              sx={{
                display: 'flex',
                flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                alignItems: 'flex-start',
                mb: 3,
                gap: 2
              }}
            >
              <Avatar sx={{ 
                background: message.role === 'user' 
                  ? 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)'
                  : 'linear-gradient(135deg, #d97706 0%, #f59e0b 100%)',
                width: 32,
                height: 32
              }}>
                {message.role === 'user' ? 
                  <PersonIcon fontSize="small" /> : 
                  <SmartToyIcon fontSize="small" />
                }
              </Avatar>
              
              <Box
                className={`message-container ${message.role}-message`}
                sx={{
                  position: 'relative',
                  '&::before': message.role === 'user' ? {
                    content: '""',
                    position: 'absolute',
                    top: 10,
                    right: -8,
                    width: 0,
                    height: 0,
                    borderLeft: '8px solid',
                    borderLeftColor: message.role === 'user' ? 'primary.main' : 'transparent',
                    borderTop: '8px solid transparent',
                    borderBottom: '8px solid transparent',
                  } : {
                    content: '""',
                    position: 'absolute',
                    top: 10,
                    left: -8,
                    width: 0,
                    height: 0,
                    borderRight: '8px solid',
                    borderRightColor: 'background.paper',
                    borderTop: '8px solid transparent',
                    borderBottom: '8px solid transparent',
                  }
                }}
              >
                {message.role === 'assistant' ? (
                  <ReactMarkdown 
                    components={{
                      h1: ({node, ...props}) => <Typography variant="h6" component="h1" sx={{ fontWeight: 600, mb: 1 }} {...props} />,
                      h2: ({node, ...props}) => <Typography variant="subtitle1" component="h2" sx={{ fontWeight: 600, mb: 1 }} {...props} />,
                      h3: ({node, ...props}) => <Typography variant="subtitle2" component="h3" sx={{ fontWeight: 600, mb: 1 }} {...props} />,
                      p: ({node, ...props}) => <Typography variant="body2" sx={{ mb: 1, lineHeight: 1.6 }} {...props} />,
                      ul: ({node, ...props}) => <Box component="ul" sx={{ pl: 2, mb: 1 }} {...props} />,
                      ol: ({node, ...props}) => <Box component="ol" sx={{ pl: 2, mb: 1 }} {...props} />,
                      li: ({node, ...props}) => <Typography component="li" variant="body2" sx={{ mb: 0.5 }} {...props} />,
                      strong: ({node, ...props}) => <Typography component="strong" sx={{ fontWeight: 700 }} {...props} />,
                      em: ({node, ...props}) => <Typography component="em" sx={{ fontStyle: 'italic' }} {...props} />,
                      code: ({node, ...props}) => <Typography component="code" sx={{ 
                        fontFamily: 'monospace', 
                        backgroundColor: 'action.hover', 
                        px: 0.5, 
                        py: 0.25, 
                        borderRadius: 1,
                        fontSize: '0.875rem'
                      }} {...props} />
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                    {message.content}
                  </Typography>
                )}
                <Typography className="timestamp">
                  {formatTimestamp(message.timestamp)}
                </Typography>
              </Box>
            </Box>
            
            {index < messages.length - 1 && (
              <Divider sx={{ my: 2, opacity: 0.3 }} />
            )}
          </Box>
        ))}
        
        {isLoading && (
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center',
            py: 3,
            gap: 2
          }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
              AI is thinking...
            </Typography>
          </Box>
        )}
        
        <div ref={messagesEndRef} />
      </Box>
      
      <Box 
        component="form" 
        onSubmit={handleSendMessage} 
        className="input-container"
        sx={{ 
          display: 'flex', 
          alignItems: 'flex-end', 
          gap: 2,
          p: 3,
          background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
          borderTop: '2px solid',
          borderColor: 'divider'
        }}
      >
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Ask about your case, legal procedures, or upload documents..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
          multiline
          maxRows={4}
          className="legal-input"
          sx={{
            '& .MuiInputBase-root': {
              backgroundColor: 'background.paper',
              borderRadius: 3,
            },
            '& .MuiInputBase-input': {
              fontSize: '1rem',
              lineHeight: 1.5,
            }
          }}
        />
        <Button 
          variant="contained" 
          color="primary" 
          type="submit" 
          disabled={isLoading || !input.trim()}
          endIcon={<SendIcon />}
          className="legal-button-primary"
          sx={{
            minHeight: '56px',
            px: 3,
            borderRadius: 3,
            fontSize: '1rem',
            fontWeight: 600,
            boxShadow: '0 4px 14px rgba(30, 58, 138, 0.3)',
            '&:hover': {
              boxShadow: '0 6px 20px rgba(30, 58, 138, 0.4)',
            },
            '&:disabled': {
              opacity: 0.6,
            }
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </Button>
      </Box>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={onClearError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={onClearError} 
          severity="error" 
          sx={{ 
            width: '100%',
            borderRadius: 2,
            fontWeight: 500
          }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ChatPanel;
