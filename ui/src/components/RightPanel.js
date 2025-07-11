import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Button, 
  Card, 
  CardContent, 
  CardActions,
  Box,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Avatar,
  Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DescriptionIcon from '@mui/icons-material/Description';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import GavelIcon from '@mui/icons-material/Gavel';
import LibraryBooksIcon from '@mui/icons-material/LibraryBooks';
import LinkIcon from '@mui/icons-material/Link';
import Api from '../services/Api';

const default_legal_advice = 
  'Legal processes often involve steps such as identifying the appropriate forms, ' +
  'completing and submitting required documents, understanding deadlines, and following ' +
  'ethical guidelines for honesty and confidentiality. Whether you are filing a case, ' +
  'responding to legal notices, or simply filling out forms, it is important to ensure ' +
  'accuracy and completeness. Procedures and requirements can vary by jurisdiction and case type.';

const default_resources = 'Relevant web resources will appear here during your conversation.';

const RightPanel = ({ currentCase, onSetLegalReferences, legalAdvice, messageCount }) => {
  const [loading, setLoading] = useState(false);
  // Initialize with default advice
  const [internalLegalAdvice, setInternalLegalAdvice] = useState({
    //title: 'General Legal Information',
    content: default_legal_advice,
    resources: default_resources
  });
  const [hasNewAdvice, setHasNewAdvice] = useState(false);
  const [lastCheckedMessageCount, setLastCheckedMessageCount] = useState(0);
  
  useEffect(() => {
    if (legalAdvice) {
      setInternalLegalAdvice(legalAdvice);
      setHasNewAdvice(true);
      
      // After 5 seconds, remove the "new" indicator
      const timer = setTimeout(() => {
        setHasNewAdvice(false);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [legalAdvice]);

  useEffect(() => {
    // Reset hasNewAdvice when case changes
    if (currentCase) {
      setHasNewAdvice(false);
    }
  }, [currentCase]);
  
  // Check for new advice every 10 messages
  useEffect(() => {
    const fetchAdviceIfNeeded = async () => {
      // Only fetch if message count is divisible by 10 and has changed since last check
      console.log(`Message count: ${messageCount}, Last checked: ${lastCheckedMessageCount}`);
      if (messageCount > 0 && messageCount % 10 === 0 && messageCount !== lastCheckedMessageCount) {
        try {
          const response = await Api.getGeneralAdvice();
          if (response.success) {
            setInternalLegalAdvice({
              title: 'Updated Legal Guidance',
              content: response.advice,
              resources: response.resources,
              titles: response.titles
            });
            setHasNewAdvice(true);
            
            // Update last checked count
            setLastCheckedMessageCount(messageCount);
            
            // After 5 seconds, remove the "new" indicator
            setTimeout(() => {
              setHasNewAdvice(false);
            }, 5000);
          }
        } catch (error) {
          console.error('Error fetching general advice:', error);
        }
      }
    };
    
    fetchAdviceIfNeeded();
  }, [messageCount, lastCheckedMessageCount]);

  const handleSetLegalReferences = async () => {
    setLoading(true);
    await onSetLegalReferences();
    setLoading(false);
    setInternalLegalAdvice({
      title: 'Legal References Set',
      content: `Legal references for ${currentCase?.jurisdiction_code || 'this case'} have been loaded and are ready for use in answering questions. Ask questions about legal considerations related to your case in the main chat panel.`
    });
  };

  const getPlaceholderAdvice = () => {
    if (!currentCase) return null;
    
    // Return different placeholder advice based on case type or jurisdiction
    if (currentCase.jurisdiction_code?.toLowerCase()?.includes('ontario') || 
        currentCase.jurisdiction_code?.toLowerCase()?.includes('mississauga')) {
      return {
        title: 'Ontario Legal Considerations',
        content: 'For Ontario-based cases, consider the following:\n\n1. The Residential Tenancies Act governs landlord-tenant relationships\n2. Small claims court can handle disputes up to $35,000\n3. Legal Aid Ontario may provide assistance for qualifying individuals',
        resources: 'Relevant web resources will appear here during your conversation.'
      };
    } else {
      return {
        //',
        content: default_legal_advice,
        resources: default_resources
      };
    }
  };

  return (
    <Box sx={{ p: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Avatar sx={{ 
          mr: 2, 
          background: 'linear-gradient(135deg, #d97706 0%, #f59e0b 100%)',
          width: 32,
          height: 32
        }}>
          <GavelIcon fontSize="small" />
        </Avatar>
        <Typography variant="h6" className="panel-title" sx={{
          color: 'primary.main',
          fontWeight: 700,
          fontSize: '1.125rem'
        }}>
          Smart Legal Guidance
        </Typography>
      </Box>
      
      {currentCase ? (
        <>
          <Card 
            variant="outlined" 
            className="legal-card"
            sx={{ 
              mb: 3,
              background: hasNewAdvice 
                ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%)'
                : 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
              border: hasNewAdvice ? '2px solid' : '1px solid',
              borderColor: hasNewAdvice ? 'primary.light' : 'divider',
              borderRadius: 3,
              transition: 'all 0.3s ease',
              position: 'relative',
              overflow: 'visible'
            }}
          >
            {hasNewAdvice && (
              <Chip
                label="Updated"
                size="small"
                color="primary"
                sx={{
                  position: 'absolute',
                  top: -8,
                  right: 16,
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  zIndex: 1
                }}
              />
            )}
            
            <CardContent sx={{ pb: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ 
                  mr: 2, 
                  background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
                  width: 28,
                  height: 28
                }}>
                  <LibraryBooksIcon fontSize="small" />
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                  {internalLegalAdvice?.title || getPlaceholderAdvice()?.title || 'Legal Guidance'}
                </Typography>
                {hasNewAdvice && (
                  <AutorenewIcon color="primary" sx={{ ml: 1, fontSize: 20 }} />
                )}
              </Box>
              
              <Typography 
                variant="body2" 
                component="div" 
                sx={{ 
                  whiteSpace: 'pre-line',
                  lineHeight: 1.6,
                  color: 'text.secondary',
                  mb: 2
                }}
              >
                {internalLegalAdvice?.content || getPlaceholderAdvice()?.content}
              </Typography>
            </CardContent>
            
            <CardActions sx={{ px: 2, pb: 2 }}>
              <Button
                variant="contained"
                size="small"
                color="primary"
                onClick={handleSetLegalReferences}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <DescriptionIcon />}
                sx={{
                  borderRadius: 2,
                  px: 2,
                  fontSize: '0.875rem',
                  fontWeight: 600,
                  textTransform: 'none',
                  background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
                  color: 'white',
                  boxShadow: '0 4px 12px rgba(30, 58, 138, 0.2)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #1e40af 0%, #2563eb 100%)',
                    boxShadow: '0 6px 16px rgba(30, 58, 138, 0.3)',
                    transform: 'translateY(-1px)',
                  },
                  '&:disabled': {
                    background: '#9ca3af',
                    color: 'white',
                    opacity: 0.7,
                  }
                }}
              >
                {loading ? 'Loading References...' : 'Load Legal References'}
              </Button>
            </CardActions>
          </Card>
          
          {/* Legal Resources Section */}
          <Accordion 
            className="legal-accordion"
            elevation={0}
          >
            <AccordionSummary 
              expandIcon={<ExpandMoreIcon />}
              className="legal-accordion-summary"
              sx={{
                '& .MuiAccordionSummary-content': {
                  alignItems: 'center'
                }
              }}
            >
              <LinkIcon sx={{ mr: 1.5, color: 'primary.main' }} />
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                Live Resources
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0 }}>
              {internalLegalAdvice?.resources && internalLegalAdvice.resources !== default_resources ? (
                <Card 
                  variant="outlined"
                  sx={{ 
                    background: 'linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%)',
                    border: '1px solid',
                    borderColor: 'info.light',
                    borderRadius: 2
                  }}
                >
                  <CardContent sx={{ py: 2 }}>
                    {internalLegalAdvice.resources && Array.isArray(internalLegalAdvice.resources) ? (
                      <Box>
                        {internalLegalAdvice.resources.map((resource, index) => {
                          // Use title if available, otherwise use the resource URL itself
                          const linkText = internalLegalAdvice.titles && 
                                         Array.isArray(internalLegalAdvice.titles) && 
                                         internalLegalAdvice.titles[index] 
                                         ? internalLegalAdvice.titles[index] 
                                         : resource;
                          
                          return (
                            <Box key={index} sx={{ mb: 1.5 }}>
                              <Typography
                                component="a"
                                href={resource}
                                target="_blank"
                                rel="noopener noreferrer"
                                variant="body2"
                                sx={{
                                  color: 'primary.main',
                                  textDecoration: 'none',
                                  fontWeight: 500,
                                  display: 'flex',
                                  alignItems: 'center',
                                  '&:hover': {
                                    textDecoration: 'underline',
                                    color: 'primary.dark'
                                  }
                                }}
                              >
                                <LinkIcon sx={{ mr: 1, fontSize: 16 }} />
                                {linkText}
                              </Typography>
                            </Box>
                          );
                        })}
                      </Box>
                    ) : (
                      <Typography 
                        variant="body2" 
                        component="div" 
                        sx={{ 
                          color: 'text.secondary',
                          lineHeight: 1.6,
                          whiteSpace: 'pre-line'
                        }}
                      >
                        {internalLegalAdvice.resources}
                      </Typography>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Box sx={{ 
                  textAlign: 'center', 
                  py: 3, 
                  background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                  borderRadius: 2,
                  border: '2px dashed',
                  borderColor: 'divider'
                }}>
                  <LinkIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
                  <Typography variant="body2" color="text.secondary">
                    Resources will appear as you interact with the AI
                  </Typography>
                </Box>
              )}
            </AccordionDetails>
          </Accordion>
        </>
      ) : (
        <Box sx={{ 
          textAlign: 'center', 
          py: 6,
          background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
          borderRadius: 3,
          border: '2px dashed',
          borderColor: 'divider'
        }}>
          <Avatar sx={{ 
            mx: 'auto',
            mb: 2,
            width: 56,
            height: 56,
            background: 'linear-gradient(135deg, #6b7280 0%, #9ca3af 100%)'
          }}>
            <GavelIcon sx={{ fontSize: 28 }} />
          </Avatar>
          <Typography variant="h6" sx={{ mb: 1, fontWeight: 600, color: 'text.primary' }}>
            Legal Guidance Ready
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Please select or create a case to begin receiving legal guidance
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default RightPanel;
