#!/bin/bash

# Package AML Dashboard for Stakeholders
# This script creates a clean package for distribution

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Packaging AML Dashboard for Stakeholders${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Create package directory
PACKAGE_DIR="aml-dashboard-stakeholder-package"
echo -e "${YELLOW}📁 Creating package directory...${NC}"
rm -rf $PACKAGE_DIR
mkdir -p $PACKAGE_DIR

# Copy essential files
echo -e "${YELLOW}📋 Copying essential files...${NC}"
cp app.py $PACKAGE_DIR/
cp requirements.txt $PACKAGE_DIR/
cp Dockerfile $PACKAGE_DIR/
cp docker-compose.yml $PACKAGE_DIR/
cp .dockerignore $PACKAGE_DIR/
cp run-aml-dashboard.sh $PACKAGE_DIR/
cp stop-aml-dashboard.sh $PACKAGE_DIR/
cp README-STAKESHOLDER.md $PACKAGE_DIR/

# Create data directory
mkdir -p $PACKAGE_DIR/data

# Create sample data file
echo -e "${YELLOW}📊 Creating sample data file...${NC}"
cat > $PACKAGE_DIR/data/sample_transactions.csv << 'EOF'
customer_no,CustomerName,transfer_type,amount,beneficiary_name,createdDateTime,reference_no
CUST001,John Doe,INTERNATIONAL_PAYMENT,5000.00,Beneficiary A,2024-01-15 10:30:00,REF001
CUST002,Jane Smith,TOP-UP,1000.00,,2024-01-15 11:15:00,REF002
CUST003,Bob Johnson,INTERNATIONAL_PAYMENT,7500.00,Beneficiary B,2024-01-15 12:00:00,REF003
CUST001,John Doe,TOP-UP,2000.00,,2024-01-15 13:45:00,REF004
CUST004,Alice Brown,INTERNATIONAL_PAYMENT,3000.00,Beneficiary C,2024-01-15 14:20:00,REF005
CUST002,Jane Smith,INTERNATIONAL_PAYMENT,4500.00,Beneficiary D,2024-01-15 15:10:00,REF006
CUST005,Charlie Wilson,TOP-UP,1500.00,,2024-01-15 16:30:00,REF007
CUST001,John Doe,INTERNATIONAL_PAYMENT,8000.00,Beneficiary E,2024-01-15 17:15:00,REF008
CUST003,Bob Johnson,TOP-UP,3000.00,,2024-01-15 18:00:00,REF009
CUST004,Alice Brown,INTERNATIONAL_PAYMENT,6000.00,Beneficiary F,2024-01-15 19:30:00,REF010
EOF

# Create quick start guide
echo -e "${YELLOW}📖 Creating quick start guide...${NC}"
cat > $PACKAGE_DIR/QUICK-START.md << 'EOF'
# 🚀 Quick Start Guide

## 1. Install Docker
- **Windows/Mac**: Download Docker Desktop from https://www.docker.com/products/docker-desktop
- **Linux**: Run `sudo apt-get install docker.io`

## 2. Start Docker
- **Windows/Mac**: Open Docker Desktop
- **Linux**: Run `sudo systemctl start docker`

## 3. Run the Dashboard
```bash
# Make scripts executable (Linux/Mac only)
chmod +x run-aml-dashboard.sh stop-aml-dashboard.sh

# Start the dashboard
./run-aml-dashboard.sh
```

## 4. Access the Dashboard
- Open your web browser
- Go to: http://localhost:8501
- Upload your CSV files or use the sample data

## 5. Stop the Dashboard
```bash
./stop-aml-dashboard.sh
```

## 📁 Sample Data
- A sample CSV file is included in the `data` folder
- You can replace it with your own transaction data
- The dashboard will automatically detect CSV files

## 🔧 Troubleshooting
- If Docker isn't running: Start Docker Desktop
- If port 8501 is busy: Stop other applications using that port
- For more help: Read README-STAKESHOLDER.md
EOF

# Create a simple installer script
echo -e "${YELLOW}🔧 Creating installer script...${NC}"
cat > $PACKAGE_DIR/install.sh << 'EOF'
#!/bin/bash

echo "🚀 AML Dashboard Installer"
echo "=========================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo ""
    echo "Please install Docker first:"
    echo "• Windows/Mac: https://www.docker.com/products/docker-desktop"
    echo "• Linux: sudo apt-get install docker.io"
    echo ""
    echo "After installing Docker, run this script again."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running."
    echo ""
    echo "Please start Docker:"
    echo "• Windows/Mac: Open Docker Desktop"
    echo "• Linux: sudo systemctl start docker"
    echo ""
    echo "After starting Docker, run this script again."
    exit 1
fi

echo "✅ Docker is installed and running"
echo ""

# Make scripts executable
chmod +x run-aml-dashboard.sh stop-aml-dashboard.sh

echo "✅ Installation complete!"
echo ""
echo "To start the dashboard, run:"
echo "./run-aml-dashboard.sh"
echo ""
echo "To stop the dashboard, run:"
echo "./stop-aml-dashboard.sh"
EOF

chmod +x $PACKAGE_DIR/install.sh

# Create a Windows batch file
echo -e "${YELLOW}🪟 Creating Windows batch file...${NC}"
cat > $PACKAGE_DIR/start-dashboard.bat << 'EOF'
@echo off
echo 🚀 Starting AML Dashboard...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running.
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Build and start the dashboard
echo 🔨 Building dashboard...
docker-compose build

echo 🚀 Starting dashboard...
docker-compose up -d

echo.
echo ✅ AML Dashboard is starting...
echo.
echo 📋 Access Information:
echo    • Local URL: http://localhost:8501
echo    • Network URL: http://%COMPUTERNAME%:8501
echo.
echo 💡 Tip: Open your web browser and go to http://localhost:8501
echo.
pause
EOF

# Create a Windows stop script
cat > $PACKAGE_DIR/stop-dashboard.bat << 'EOF'
@echo off
echo 🛑 Stopping AML Dashboard...
echo.

docker-compose down

echo ✅ AML Dashboard has been stopped!
echo.
echo 📋 To start again, run: start-dashboard.bat
echo.
pause
EOF

# Create package info
echo -e "${YELLOW}📋 Creating package info...${NC}"
cat > $PACKAGE_DIR/PACKAGE-INFO.txt << 'EOF'
AML Dashboard - Stakeholder Package
==================================

This package contains everything needed to run the AML Dashboard on your laptop.

Contents:
• app.py - Main application
• requirements.txt - Python dependencies
• Dockerfile - Container configuration
• docker-compose.yml - Easy deployment
• run-aml-dashboard.sh - Start script (Linux/Mac)
• stop-aml-dashboard.sh - Stop script (Linux/Mac)
• start-dashboard.bat - Start script (Windows)
• stop-dashboard.bat - Stop script (Windows)
• install.sh - Installation script (Linux/Mac)
• README-STAKESHOLDER.md - Detailed guide
• QUICK-START.md - Quick start guide
• data/ - Sample data folder

System Requirements:
• Docker Desktop (Windows/Mac) or Docker (Linux)
• 4GB RAM minimum, 8GB recommended
• 2GB free disk space
• Modern web browser

Quick Start:
1. Install Docker Desktop
2. Run install.sh (Linux/Mac) or start-dashboard.bat (Windows)
3. Open http://localhost:8501 in your browser

For detailed instructions, see README-STAKESHOLDER.md
EOF

# Create the final package
echo -e "${YELLOW}📦 Creating final package...${NC}"
tar -czf aml-dashboard-stakeholder-package.tar.gz $PACKAGE_DIR

echo ""
echo -e "${GREEN}✅ Package created successfully!${NC}"
echo ""
echo -e "${BLUE}📦 Package Information:${NC}"
echo "   • Package name: aml-dashboard-stakeholder-package.tar.gz"
echo "   • Size: $(du -h aml-dashboard-stakeholder-package.tar.gz | cut -f1)"
echo "   • Contents: $PACKAGE_DIR/"
echo ""
echo -e "${BLUE}📋 Distribution Instructions:${NC}"
echo "   1. Send the .tar.gz file to stakeholders"
echo "   2. They extract it: tar -xzf aml-dashboard-stakeholder-package.tar.gz"
echo "   3. They run: cd aml-dashboard-stakeholder-package && ./install.sh"
echo "   4. They start: ./run-aml-dashboard.sh"
echo ""
echo -e "${GREEN}🎉 Package ready for distribution!${NC}" 