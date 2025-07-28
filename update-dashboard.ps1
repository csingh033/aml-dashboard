# AML Dashboard Update Script for Windows
# Run this script in PowerShell as Administrator

Write-Host "🚀 AML Dashboard Update Script" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Step 1: Pull latest changes
Write-Host "`n📥 Step 1: Pulling latest changes from repository..." -ForegroundColor Yellow
git pull origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Repository updated successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to pull changes. Please check your git status." -ForegroundColor Red
    exit 1
}

# Step 2: Stop existing containers
Write-Host "`n🔧 Step 2: Stopping existing containers..." -ForegroundColor Yellow
docker stop $(docker ps -q) 2>$null
Write-Host "✅ Containers stopped!" -ForegroundColor Green

# Step 3: Build new Docker image
Write-Host "`n🏗️ Step 3: Building new Docker image with updated dependencies..." -ForegroundColor Yellow
docker build -f docker/Dockerfile -t aml-dashboard .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Docker image built successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Docker build failed. Please check the error messages above." -ForegroundColor Red
    exit 1
}

# Step 4: Run the dashboard
Write-Host "`n🚀 Step 4: Starting the AML Dashboard..." -ForegroundColor Yellow
Write-Host "The dashboard will be available at: http://localhost:8502" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the dashboard when needed." -ForegroundColor Cyan
Write-Host "`nStarting dashboard..." -ForegroundColor Green

# Run the dashboard with correct port mapping
docker run -p 8502:8501 aml-dashboard 