#!/bin/bash

#****************************************************************************=
# Legal RAG Pipeline - Application Startup Script
#****************************************************************************=

# Display header
echo "*************************************************"
echo "Legal RAG Pipeline - Starting Application Services"
echo "*************************************************"

# Configure environment for development
export FLASK_APP=api/app.py
export FLASK_ENV=development


# DEPENDENCY CHECKS


echo "Performing dependency checks..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    echo "Please install Python 3.11+ from https://python.org"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required but not installed"
    echo "Please install Node.js from https://nodejs.org"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is required but not installed"
    echo "npm should be included with Node.js installation"
    exit 1
fi

echo "All required dependencies found"


# PACKAGE INSTALLATION


# Install Python dependencies if needed
echo "Checking Python dependencies..."
pip install flask flask-cors python-dotenv

# Install Node.js dependencies if needed
echo "Checking Node.js dependencies..."
if [ ! -d "ui/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd ui && npm install && cd ..
fi


# SERVICE STARTUP


# Start backend server
echo "Starting Flask API server..."
python3 api/app.py &
API_PID=$!
echo "Flask API server started with PID: $API_PID"

# Wait for backend to initialize
sleep 2

# Start frontend server
echo "Starting React development server..."
cd ui && BROWSER=none DANGEROUSLY_DISABLE_HOST_CHECK=true npm start &
UI_PID=$!
echo "React development server started with PID: $UI_PID"


# APPLICATION STATUS


echo ""
echo "*************************************************"
echo "Application services successfully started"
echo "*************************************************"
echo "Backend API:  http://localhost:5000"
echo "Frontend UI:  http://localhost:3000"
echo "*************************************************"
echo "Press Ctrl+C to stop all services"
echo ""


# CLEANUP AND SIGNAL HANDLING


# Function to handle graceful cleanup on exit
function cleanup() {
    echo ""
    echo "Stopping services..."
    
    # Terminate backend process
    if kill -0 $API_PID 2>/dev/null; then
        echo "Stopping Flask API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null
    fi
    
    # Terminate frontend process
    if kill -0 $UI_PID 2>/dev/null; then
        echo "Stopping React development server (PID: $UI_PID)..."
        kill $UI_PID 2>/dev/null
    fi
    
    echo "All services stopped successfully"
    exit 0
}

# Set up signal traps for graceful shutdown
trap cleanup SIGINT SIGTERM EXIT

# Keep script running until interrupted
echo "Monitoring services... (Use Ctrl+C to stop)"
wait
