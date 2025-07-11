import React from 'react';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  Box, 
  IconButton,
  Chip,
  Avatar
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import SaveIcon from '@mui/icons-material/Save';
import BalanceIcon from '@mui/icons-material/Balance';
import MenuIcon from '@mui/icons-material/Menu';
import ClearIcon from '@mui/icons-material/Clear';

const Header = ({ isInitialized, currentCase, currentClient, onOpenSetup, onSaveAll, onToggleLeftPanel, onClearSession, leftPanelVisible }) => {
  return (
    <AppBar 
      position="fixed" 
      elevation={0}
      sx={{
        background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <Toolbar sx={{ px: 3 }}>
        <IconButton
          color="inherit"
          aria-label="toggle left panel"
          onClick={onToggleLeftPanel}
          edge="start"
          sx={{ 
            mr: 2,
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            }
          }}
        >
          <MenuIcon />
        </IconButton>
        
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 3 }}>
          <Avatar sx={{ 
            mr: 2, 
            background: 'linear-gradient(135deg, #d97706 0%, #f59e0b 100%)',
            width: 40,
            height: 40
          }}>
            <BalanceIcon />
          </Avatar>
          <Box>
            <Typography 
              variant="h5" 
              component="div" 
              sx={{ 
                fontWeight: 700,
                letterSpacing: '-0.025em',
                fontSize: '1.5rem',
                lineHeight: 1
              }}
            >
              LegalNexus
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                opacity: 0.8,
                fontSize: '0.75rem',
                letterSpacing: '0.05em',
                textTransform: 'uppercase'
              }}
            >
              AI Legal Research Assistant
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ flexGrow: 1 }} />
        
        {currentCase && (
          <Box sx={{ display: 'flex', alignItems: 'center', mr: 3, gap: 1.5 }}>
            <Chip 
              label={`Case: ${currentCase.name}`} 
              variant="filled"
              sx={{ 
                background: 'rgba(255, 255, 255, 0.2)', 
                color: 'white',
                borderColor: 'rgba(255, 255, 255, 0.3)',
                backdropFilter: 'blur(10px)',
                fontWeight: 600,
                fontSize: '0.875rem',
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.25)',
                }
              }} 
            />
            
            {currentClient && (
              <Chip 
                label={`Client: ${currentClient.name}`}
                variant="filled" 
                sx={{ 
                  background: 'rgba(217, 119, 6, 0.2)',
                  color: 'white',
                  borderColor: 'rgba(217, 119, 6, 0.3)',
                  backdropFilter: 'blur(10px)',
                  fontWeight: 600,
                  fontSize: '0.875rem',
                  '&:hover': {
                    background: 'rgba(217, 119, 6, 0.3)',
                  }
                }}
              />
            )}
          </Box>
        )}
        
        {isInitialized && (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button 
              color="inherit" 
              startIcon={<SaveIcon />}
              onClick={onSaveAll}
              sx={{ 
                fontWeight: 600,
                textTransform: 'none',
                borderRadius: 2,
                px: 2,
                '&:hover': {
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                }
              }}
            >
              Save
            </Button>
            
            {(currentCase || currentClient) && (
              <Button 
                color="inherit" 
                startIcon={<ClearIcon />}
                onClick={onClearSession}
                sx={{ 
                  fontWeight: 600,
                  textTransform: 'none',
                  borderRadius: 2,
                  px: 2,
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  }
                }}
              >
                Clear Session
              </Button>
            )}
          </Box>
        )}
        
        <IconButton 
          color="inherit" 
          edge="end" 
          onClick={onOpenSetup}
          sx={{ 
            ml: 2,
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
            }
          }}
          aria-label="settings"
        >
          <SettingsIcon />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
