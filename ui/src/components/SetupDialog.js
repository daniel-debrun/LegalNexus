import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  CircularProgress,
  Alert,
  Box,
  Avatar,
  Divider
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import SmartToyIcon from '@mui/icons-material/SmartToy';

const SetupDialog = ({ open, onClose, onInitialize, isLoading, error, onClearError }) => {
  const [modelName, setModelName] = useState('gemma3:4b');

  const handleSubmit = (e) => {
    e.preventDefault();
    onInitialize(modelName);
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="sm" 
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 3,
          background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
        }
      }}
    >
      <Box sx={{ textAlign: 'center', pt: 3, pb: 1 }}>
        <Avatar sx={{ 
          mx: 'auto',
          mb: 2,
          width: 64,
          height: 64,
          background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)'
        }}>
          <SettingsIcon sx={{ fontSize: 32 }} />
        </Avatar>
        <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary', mb: 1 }}>
          Configure Legal AI Assistant
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Set up your intelligent legal research companion
        </Typography>
      </Box>
      
      <Divider sx={{ my: 2 }} />
      
      <form onSubmit={handleSubmit}>
        <DialogContent sx={{ px: 4 }}>
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ 
                mr: 2, 
                background: 'linear-gradient(135deg, #d97706 0%, #f59e0b 100%)',
                width: 32,
                height: 32
              }}>
                <SmartToyIcon fontSize="small" />
              </Avatar>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                AI Model Configuration
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary" paragraph>
              Choose the AI model that will power your legal assistant. The default model is optimized for legal research and analysis.
            </Typography>
          </Box>
          
          <TextField
            label="AI Model Name"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            fullWidth
            margin="normal"
            helperText="Recommended: gemma3:4b for optimal performance"
            required
            className="legal-input"
            sx={{
              '& .MuiInputLabel-root': {
                fontWeight: 500
              },
              '& .MuiFormHelperText-root': {
                fontSize: '0.75rem',
                color: 'text.secondary'
              }
            }}
          />
          
          {error && (
            <Alert 
              severity="error" 
              onClose={onClearError}
              sx={{ 
                mt: 3,
                borderRadius: 2,
                fontWeight: 500
              }}
            >
              {error}
            </Alert>
          )}
        </DialogContent>
        
        <DialogActions sx={{ px: 4, pb: 3, justifyContent: 'center' }}>
          <Button 
            type="submit" 
            variant="contained" 
            color="primary"
            disabled={isLoading || !modelName}
            startIcon={isLoading ? <CircularProgress size={16} color="inherit" /> : <SmartToyIcon />}
            className="legal-button-primary"
            sx={{
              px: 4,
              py: 1.5,
              fontSize: '1rem',
              fontWeight: 600,
              borderRadius: 2,
              minWidth: 200,
              boxShadow: '0 4px 14px rgba(30, 58, 138, 0.3)',
              '&:hover': {
                boxShadow: '0 6px 20px rgba(30, 58, 138, 0.4)',
              }
            }}
          >
            {isLoading ? 'Initializing AI...' : 'Initialize Assistant'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default SetupDialog;
