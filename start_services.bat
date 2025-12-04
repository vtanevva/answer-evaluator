@echo off
echo 🚀 Starting Answer Evaluator Services...
echo.

echo 📋 Starting Backend (FastAPI with Uvicorn)...
start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo ⏳ Waiting 3 seconds for backend to initialize...
timeout /t 3 /nobreak > nul

echo 🎨 Starting Frontend (React)...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo ✅ Both services are starting!
echo 📊 Backend: http://localhost:8000
echo 🎨 Frontend: http://localhost:3000
echo.
echo Press any key to exit this window...
pause > nul
