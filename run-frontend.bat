@echo off
cd /d "%~dp0frontend"
if not exist node_modules (
  echo Installing frontend dependencies...
  npm install
)
echo Starting BALMORES STRUCTURAL frontend at http://localhost:3000
npm run dev
