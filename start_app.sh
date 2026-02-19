#!/bin/bash

# Start Backend
echo "Starting Backend Server on port 8000..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py &
BACKEND_PID=$!
cd ..

# Start Frontend
echo "Starting Frontend Server on port 3000..."
npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT

wait
