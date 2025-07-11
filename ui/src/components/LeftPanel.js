import React, { useState, useEffect } from 'react';
import {
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  Autocomplete,
  Avatar,
  Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import UploadIcon from '@mui/icons-material/CloudUpload';
import FolderIcon from '@mui/icons-material/Folder';
import PersonIcon from '@mui/icons-material/Person';
import DescriptionIcon from '@mui/icons-material/Description';
import BusinessIcon from '@mui/icons-material/Business';

// Import jurisdiction codes and names
import jurisdictionCodes from '../jurisdiction_code_names.json';
import Api from '../services/Api';

// Convert jurisdiction codes to options for the dropdown
const jurisdictionOptions = Object.entries(jurisdictionCodes).map(([code, name]) => ({
  code,
  name
}));

const LeftPanel = ({
  currentCase,
  currentClient,
  documents,
  onLoadCase,
  onCreateCase,
  onLoadClient,
  onCreateClient,
  onUploadDocument,
  onDeleteDocument,
  isLoading
}) => {
  // State for dialogs
  const [openCaseDialog, setOpenCaseDialog] = useState(false);
  const [openClientDialog, setOpenClientDialog] = useState(false);
  const [openUploadDialog, setOpenUploadDialog] = useState(false);
  
  // State for forms
  const [caseForm, setCaseForm] = useState({
    name: '',
    jurisdiction_code: '',
    notes: '',
    case_type: 'Civil',
    case_status: 'Open'
  });
  
  const [selectedJurisdiction, setSelectedJurisdiction] = useState(null);
  
  const [clientForm, setClientForm] = useState({
    name: '',
    email: '',
    phone: '',
    address: '',
    date_of_birth: '',
    gender: '',
    occupation: '',
    notes: ''
  });
  
  const [uploadForm, setUploadForm] = useState({
    filePaths: '',
    urls: ''
  });

  // State for autocomplete options
  const [caseOptions, setCaseOptions] = useState([]);
  const [clientOptions, setClientOptions] = useState([]);
  const [selectedCase, setSelectedCase] = useState(null);
  const [selectedClient, setSelectedClient] = useState(null);

  // State for tracking edit mode
  const [isEditingCase, setIsEditingCase] = useState(false);
  const [isEditingClient, setIsEditingClient] = useState(false);

  // State for accordion expansion
  const [caseAccordionExpanded, setCaseAccordionExpanded] = useState(true);
  const [clientAccordionExpanded, setClientAccordionExpanded] = useState(true);

  // Handle form changes
  const handleCaseFormChange = (e) => {
    setCaseForm({
      ...caseForm,
      [e.target.name]: e.target.value
    });
  };
  
  const handleClientFormChange = (e) => {
    setClientForm({
      ...clientForm,
      [e.target.name]: e.target.value
    });
  };
  
  const handleUploadFormChange = (e) => {
    setUploadForm({
      ...uploadForm,
      [e.target.name]: e.target.value
    });
  };
  
  // Handle form submissions
  const handleCaseSubmit = (e) => {
    e.preventDefault();
    const caseData = {
      ...caseForm,
      jurisdiction_code: selectedJurisdiction?.code || ''
    };
    
    // Include case ID when editing
    if (isEditingCase && currentCase && currentCase.id) {
      caseData.case_id = currentCase.id;
    }
    
    onCreateCase(caseData);
    setOpenCaseDialog(false);
    setIsEditingCase(false);
    setCaseAccordionExpanded(false); // Collapse the accordion
    setCaseForm({
      name: '',
      jurisdiction_code: '',
      notes: '',
      case_type: 'Civil',
      case_status: 'Open'
    });
    setSelectedJurisdiction(null);
  };
  
  const handleClientSubmit = (e) => {
    e.preventDefault();
    
    // Structure the client data according to the Client model
    const clientData = {
      name: clientForm.name,
      notes: clientForm.notes,
      client_details: {
        email: clientForm.email,
        phone: clientForm.phone,
        address: clientForm.address,
        date_of_birth: clientForm.date_of_birth,
        gender: clientForm.gender,
        occupation: clientForm.occupation
      }
    };
    
    // Include client ID when editing
    if (isEditingClient && currentClient && currentClient.id) {
      clientData.client_id = currentClient.id;
    }
    
    onCreateClient(clientData);
    setOpenClientDialog(false);
    setIsEditingClient(false);
    setClientAccordionExpanded(false); // Collapse the accordion
    setClientForm({
      name: '',
      email: '',
      phone: '',
      address: '',
      date_of_birth: '',
      gender: '',
      occupation: '',
      notes: ''
    });
  };
  
  const handleUploadSubmit = (e) => {
    e.preventDefault();
    const filePaths = uploadForm.filePaths.split('\n').filter(path => path.trim());
    const urls = uploadForm.urls.split('\n').filter(url => url.trim());
    onUploadDocument(filePaths, urls);
    setOpenUploadDialog(false);
    setUploadForm({
      filePaths: '',
      urls: ''
    });
  };
  
  const handleSearchCase = () => {
    if (selectedCase) {
      onLoadCase(selectedCase.name);
      setSelectedCase(null);
    }
  };
  
  const handleSearchClient = () => {
    if (selectedClient) {
      onLoadClient(selectedClient.name);
      setSelectedClient(null);
    }
  };

  // Functions to handle editing existing case/client
  const handleEditCase = () => {
    if (currentCase) {
      // Pre-populate the form with current case data
      setCaseForm({
        name: currentCase.name || '',
        jurisdiction_code: currentCase.jurisdiction_code || '',
        notes: currentCase.notes || '',
        case_type: currentCase.case_type || 'Civil',
        case_status: currentCase.case_status || 'Open'
      });
      
      // Set the selected jurisdiction
      const jurisdiction = jurisdictionOptions.find(
        option => option.code === currentCase.jurisdiction_code
      );
      setSelectedJurisdiction(jurisdiction || null);
      
      setIsEditingCase(true);
      setOpenCaseDialog(true);
    }
  };

  const handleEditClient = () => {
    if (currentClient) {
      // Pre-populate the form with current client data
      setClientForm({
        name: currentClient.name || '',
        email: currentClient.email || '',
        phone: currentClient.phone || '',
        address: currentClient.address || '',
        date_of_birth: currentClient.date_of_birth || '',
        gender: currentClient.gender || '',
        occupation: currentClient.occupation || '',
        notes: currentClient.notes || ''
      });
      
      setIsEditingClient(true);
      setOpenClientDialog(true);
    }
  };

  // Functions to fetch autocomplete options
  const fetchCaseOptions = async (searchTerm = '') => {
    try {
      const response = await Api.getCases(searchTerm);
      if (response.success) {
        setCaseOptions(response.cases || []);
      }
    } catch (error) {
      console.error('Error fetching cases:', error);
      setCaseOptions([]);
    }
  };

  const fetchClientOptions = async (searchTerm = '') => {
    try {
      const response = await Api.getClients(searchTerm);
      if (response.success) {
        setClientOptions(response.clients || []);
      }
    } catch (error) {
      console.error('Error fetching clients:', error);
      setClientOptions([]);
    }
  };

  // Fetch initial case options when component mounts
  useEffect(() => {
    fetchCaseOptions();
  }, []);

  return (
    <Box sx={{ p: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Avatar sx={{ 
          mr: 2, 
          background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
          width: 32,
          height: 32
        }}>
          <BusinessIcon fontSize="small" />
        </Avatar>
        <Typography variant="h6" className="panel-title" sx={{
          color: 'primary.main',
          fontWeight: 700,
          fontSize: '1.125rem'
        }}>
          Case & Client Management
        </Typography>
      </Box>
      
      {/* Case Management */}
      <Accordion 
        expanded={caseAccordionExpanded}
        onChange={(event, isExpanded) => setCaseAccordionExpanded(isExpanded)}
        className="legal-accordion"
        elevation={0}
        sx={{ mb: 2 }}
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
          <FolderIcon sx={{ mr: 1.5, color: 'primary.main' }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Case Overview
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          {currentCase ? (
            <Card 
              variant="outlined" 
              className="legal-card"
              onClick={handleEditCase}
              sx={{ 
                mb: 2,
                background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                border: '2px solid',
                borderColor: 'primary.light',
                borderRadius: 2,
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  borderColor: 'primary.main',
                  boxShadow: '0 4px 12px rgba(30, 58, 138, 0.15)',
                  transform: 'translateY(-1px)'
                }
              }}
            >
              <CardContent sx={{ pb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ 
                    mr: 2, 
                    background: 'linear-gradient(135deg, #d97706 0%, #f59e0b 100%)',
                    width: 28,
                    height: 28
                  }}>
                    <FolderIcon fontSize="small" />
                  </Avatar>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    {currentCase.name}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Chip 
                    label={`${jurisdictionCodes[currentCase.jurisdiction_code] || currentCase.jurisdiction_code}`}
                    size="small"
                    variant="filled"
                    sx={{ 
                      alignSelf: 'flex-start',
                      background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
                      color: 'white',
                      fontWeight: 600
                    }}
                  />
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={currentCase.case_status}
                      size="small"
                      variant="outlined"
                      color={currentCase.case_status === 'Open' ? 'success' : 'default'}
                      sx={{ fontWeight: 500 }}
                    />
                    <Chip 
                      label={currentCase.case_type}
                      size="small"
                      variant="outlined"
                      sx={{ fontWeight: 500 }}
                    />
                  </Box>
                </Box>
                {currentCase.notes && (
                  <Typography variant="body2" sx={{ mt: 2, fontStyle: 'italic', color: 'text.secondary' }}>
                    {currentCase.notes.length > 50 ? `${currentCase.notes.substring(0, 50)}...` : currentCase.notes}
                  </Typography>
                )}
                <Typography variant="caption" sx={{ 
                  mt: 2, 
                  display: 'block',
                  textAlign: 'center',
                  color: 'text.disabled',
                  fontStyle: 'italic'
                }}>
                  click to edit
                </Typography>
              </CardContent>
            </Card>
          ) : (
            <Box sx={{ 
              textAlign: 'center', 
              py: 3, 
              background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
              borderRadius: 2,
              border: '2px dashed',
              borderColor: 'divider',
              mb: 2
            }}>
              <FolderIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                No case selected
              </Typography>
            </Box>
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
            <Autocomplete
              options={caseOptions}
              getOptionLabel={(option) => option.name || ''}
              value={selectedCase}
              onChange={(event, newValue) => {
                setSelectedCase(newValue);
              }}
              onInputChange={(event, newInputValue) => {
                fetchCaseOptions(newInputValue);
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search Case"
                  placeholder="Enter case name"
                  size="small"
                  className="legal-input"
                  sx={{
                    '& .MuiInputLabel-root': {
                      fontWeight: 500
                    }
                  }}
                />
              )}
              renderOption={(props, option) => (
                <Box component="li" {...props}>
                  <Box>
                    <Typography variant="body2" fontWeight={500}>
                      {option.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {option.jurisdiction_code} • {option.case_type} • {option.case_status}
                    </Typography>
                  </Box>
                </Box>
              )}
              fullWidth
              loading={isLoading}
              loadingText="Loading cases..."
              noOptionsText="No cases found"
            />
            <Button 
              variant="contained" 
              size="small" 
              onClick={handleSearchCase}
              disabled={isLoading || !selectedCase}
              className="legal-button-primary"
              sx={{ minWidth: 'auto', px: 2 }}
            >
              Load
            </Button>
          </Box>
          
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            fullWidth
            onClick={() => {
              setIsEditingCase(false);
              setOpenCaseDialog(true);
            }}
            className="legal-button-secondary"
          >
            Create New Case
          </Button>
        </AccordionDetails>
      </Accordion>
      
      {/* Client Management */}
      <Accordion 
        expanded={clientAccordionExpanded}
        onChange={(event, isExpanded) => setClientAccordionExpanded(isExpanded)}
        className="legal-accordion"
        elevation={0}
        sx={{ mb: 2 }}
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
          <PersonIcon sx={{ mr: 1.5, color: 'primary.main' }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Client Profile
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          {currentClient ? (
            <Card 
              variant="outlined" 
              className="legal-card"
              onClick={handleEditClient}
              sx={{ 
                mb: 2,
                background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
                border: '2px solid',
                borderColor: 'secondary.light',
                borderRadius: 2,
                cursor: 'pointer',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  borderColor: 'secondary.main',
                  boxShadow: '0 4px 12px rgba(16, 185, 129, 0.15)',
                  transform: 'translateY(-1px)'
                }
              }}
            >
              <CardContent sx={{ pb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Avatar sx={{ 
                    mr: 2, 
                    background: 'linear-gradient(135deg, #059669 0%, #10b981 100%)',
                    width: 28,
                    height: 28
                  }}>
                    <PersonIcon fontSize="small" />
                  </Avatar>
                  <Typography variant="h6" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    {currentClient.name}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                    <strong>Email:</strong>&nbsp;{currentClient.email}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                    <strong>Phone:</strong>&nbsp;{currentClient.phone}
                  </Typography>
                  {currentClient.address && (
                    <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                      <strong>Address:</strong>&nbsp;{currentClient.address}
                    </Typography>
                  )}
                  {currentClient.occupation && (
                    <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center' }}>
                      <strong>Occupation:</strong>&nbsp;{currentClient.occupation}
                    </Typography>
                  )}
                </Box>
                {currentClient.notes && (
                  <Typography variant="body2" sx={{ mt: 2, fontStyle: 'italic', color: 'text.secondary' }}>
                    {currentClient.notes.length > 50 ? `${currentClient.notes.substring(0, 50)}...` : currentClient.notes}
                  </Typography>
                )}
                <Typography variant="caption" sx={{ 
                  mt: 2, 
                  display: 'block',
                  textAlign: 'center',
                  color: 'text.disabled',
                  fontStyle: 'italic'
                }}>
                  click to edit
                </Typography>
              </CardContent>
            </Card>
          ) : (
            <Box sx={{ 
              textAlign: 'center', 
              py: 3, 
              background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
              borderRadius: 2,
              border: '2px dashed',
              borderColor: 'divider',
              mb: 2
            }}>
              <PersonIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                No client selected
              </Typography>
            </Box>
          )}
          
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 1 }}>
            <Autocomplete
              options={clientOptions}
              getOptionLabel={(option) => option.name || ''}
              value={selectedClient}
              onChange={(event, newValue) => {
                setSelectedClient(newValue);
              }}
              onInputChange={(event, newInputValue) => {
                fetchClientOptions(newInputValue);
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Search Client"
                  placeholder="Enter client name"
                  size="small"
                  className="legal-input"
                  sx={{
                    '& .MuiInputLabel-root': {
                      fontWeight: 500
                    }
                  }}
                />
              )}
              renderOption={(props, option) => (
                <Box component="li" {...props}>
                  <Box>
                    <Typography variant="body2" fontWeight={500}>
                      {option.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {option.email} • {option.occupation || 'No occupation'}
                    </Typography>
                  </Box>
                </Box>
              )}
              fullWidth
              loading={isLoading}
              loadingText="Loading clients..."
              noOptionsText="No clients found"
              disabled={!currentCase}
            />
            <Button 
              variant="contained" 
              size="small" 
              onClick={handleSearchClient}
              disabled={isLoading || !selectedClient || !currentCase}
              className="legal-button-primary"
              sx={{ minWidth: 'auto', px: 2 }}
            >
              Load
            </Button>
          </Box>
          
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            fullWidth
            onClick={() => {
              setIsEditingClient(false);
              setOpenClientDialog(true);
            }}
            disabled={!currentCase}
            className="legal-button-secondary"
          >
            Create New Client
          </Button>
        </AccordionDetails>
      </Accordion>
      
      {/* Documents */}
      <Accordion 
        defaultExpanded 
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
          <DescriptionIcon sx={{ mr: 1.5, color: 'primary.main' }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Documents
          </Typography>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          {documents.length > 0 ? (
            <List dense sx={{ mb: 2 }}>
              {documents.map((doc) => (
                <ListItem 
                  key={doc.id}
                  sx={{ 
                    border: '1px solid',
                    borderColor: 'divider',
                    borderRadius: 2,
                    mb: 1,
                    backgroundColor: 'background.paper',
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    }
                  }}
                >
                  <Avatar sx={{ 
                    mr: 2, 
                    background: 'linear-gradient(135deg, #6b7280 0%, #9ca3af 100%)',
                    width: 24,
                    height: 24
                  }}>
                    <DescriptionIcon fontSize="small" />
                  </Avatar>
                  <ListItemText
                    primary={
                      <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                        {doc.source_name}
                      </Typography>
                    }
                    secondary={
                      <Typography variant="caption" color="text.secondary">
                        {doc.description || doc.title || `Uploaded: ${new Date(doc.uploaded_at).toLocaleDateString()}`}
                      </Typography>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton 
                      edge="end" 
                      aria-label="delete"
                      onClick={() => onDeleteDocument(doc.id)}
                      disabled={isLoading}
                      sx={{
                        color: 'error.main',
                        '&:hover': {
                          backgroundColor: 'error.light',
                          color: 'error.contrastText'
                        }
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          ) : (
            <Box sx={{ 
              textAlign: 'center', 
              py: 3, 
              background: 'linear-gradient(135deg, #f8fafc 0%, #ffffff 100%)',
              borderRadius: 2,
              border: '2px dashed',
              borderColor: 'divider',
              mb: 2
            }}>
              <DescriptionIcon sx={{ fontSize: 48, color: 'text.disabled', mb: 1 }} />
              <Typography variant="body2" color="text.secondary">
                No documents uploaded
              </Typography>
            </Box>
          )}
          
          <Button
            variant="outlined"
            startIcon={<UploadIcon />}
            fullWidth
            onClick={() => setOpenUploadDialog(true)}
            disabled={!currentCase}
            className="legal-button-secondary"
          >
            Upload Document
          </Button>
        </AccordionDetails>
      </Accordion>
      
      {/* Dialogs */}
      
      {/* Create Case Dialog */}
      <Dialog open={openCaseDialog} onClose={() => {
        setOpenCaseDialog(false);
        setIsEditingCase(false);
      }}>
        <DialogTitle>{isEditingCase ? 'Edit Case' : 'Create New Case'}</DialogTitle>
        <form onSubmit={handleCaseSubmit}>
          <DialogContent>
            <TextField
              name="name"
              label="Case Name"
              placeholder="Enter case name"
              value={caseForm.name}
              onChange={handleCaseFormChange}
              fullWidth
              margin="normal"
              required
            />
            <Autocomplete
              options={jurisdictionOptions}
              getOptionLabel={(option) => `${option.code} - ${option.name}`}
              value={selectedJurisdiction}
              onChange={(event, newValue) => {
                setSelectedJurisdiction(newValue);
              }}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Jurisdiction"
                  placeholder="Search and select jurisdiction..."
                  margin="normal"
                  required
                />
              )}
              fullWidth
              filterOptions={(options, { inputValue }) => {
                return options.filter(option =>
                  option.name.toLowerCase().includes(inputValue.toLowerCase()) ||
                  option.code.toLowerCase().includes(inputValue.toLowerCase())
                );
              }}
            />
            <TextField
              name="case_type"
              label="Case Type"
              placeholder="e.g., Civil, Criminal, Family"
              value={caseForm.case_type}
              onChange={handleCaseFormChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="case_status"
              label="Case Status"
              placeholder="e.g., Open, Closed, Pending"
              value={caseForm.case_status}
              onChange={handleCaseFormChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="notes"
              label="Notes"
              placeholder="Additional case details..."
              value={caseForm.notes}
              onChange={handleCaseFormChange}
              fullWidth
              margin="normal"
              multiline
              rows={4}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => {
              setOpenCaseDialog(false);
              setIsEditingCase(false);
            }}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary"
              disabled={!caseForm.name || !selectedJurisdiction}
            >
              {isEditingCase ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
      
      {/* Create Client Dialog */}
      <Dialog open={openClientDialog} onClose={() => {
        setOpenClientDialog(false);
        setIsEditingClient(false);
      }}>
        <DialogTitle>{isEditingClient ? 'Edit Client' : 'Create New Client'}</DialogTitle>
        <form onSubmit={handleClientSubmit}>
          <DialogContent>
            <TextField
              name="name"
              label="Client Name"
              placeholder="Enter client full name"
              value={clientForm.name}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
              required
            />
            <TextField
              name="email"
              label="Email"
              placeholder="client@example.com"
              value={clientForm.email}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
              type="email"
              required
            />
            <TextField
              name="phone"
              label="Phone"
              placeholder="+1 (555) 123-4567"
              value={clientForm.phone}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
              required
            />
            <TextField
              name="address"
              label="Address"
              placeholder="123 Main St, City, Province"
              value={clientForm.address}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="date_of_birth"
              label="Date of Birth"
              value={clientForm.date_of_birth}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
              placeholder="YYYY-MM-DD"
            />
            <TextField
              name="gender"
              label="Gender"
              placeholder="Male/Female/Other"
              value={clientForm.gender}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="occupation"
              label="Occupation"
              placeholder="Enter profession"
              value={clientForm.occupation}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="notes"
              label="Notes"
              placeholder="Additional notes about the client..."
              value={clientForm.notes}
              onChange={handleClientFormChange}
              fullWidth
              margin="normal"
              multiline
              rows={4}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => {
              setOpenClientDialog(false);
              setIsEditingClient(false);
            }}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary"
              disabled={!clientForm.name || !clientForm.email || !clientForm.phone}
            >
              {isEditingClient ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
      
      {/* Upload Document Dialog */}
      <Dialog open={openUploadDialog} onClose={() => setOpenUploadDialog(false)}>
        <DialogTitle>Upload Document</DialogTitle>
        <form onSubmit={handleUploadSubmit}>
          <DialogContent>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Enter file paths (one per line) or URLs (one per line) to upload.
            </Typography>
            <TextField
              name="filePaths"
              label="File Paths"
              value={uploadForm.filePaths}
              onChange={handleUploadFormChange}
              fullWidth
              margin="normal"
              multiline
              rows={3}
              placeholder="/path/to/file1.pdf&#10;/path/to/file2.txt"
            />
            <Divider sx={{ my: 2 }}>OR</Divider>
            <TextField
              name="urls"
              label="URLs"
              value={uploadForm.urls}
              onChange={handleUploadFormChange}
              fullWidth
              margin="normal"
              multiline
              rows={3}
              placeholder="https://example.com/doc1&#10;https://example.com/doc2"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenUploadDialog(false)}>Cancel</Button>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary"
              disabled={!uploadForm.filePaths.trim() && !uploadForm.urls.trim()}
            >
              Upload
            </Button>
          </DialogActions>
        </form>
      </Dialog>
      
      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <CircularProgress size={24} />
        </Box>
      )}
    </Box>
  );
};

export default LeftPanel;
