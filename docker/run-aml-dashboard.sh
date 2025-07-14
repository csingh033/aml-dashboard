#!/bin/bash

# AML Dashboard - Easy Deployment Script for Stakeholders
# This script makes it easy to run the AML dashboard on any laptop

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AML Dashboard - Easy Deployment${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    echo ""
    echo "📋 Installation Instructions:"
    echo "   • Windows/Mac: Download Docker Desktop from https://www.docker.com/products/docker-desktop"
    echo "   • Linux: Run 'sudo apt-get install docker.io' or 'sudo yum install docker'"
    echo ""
    echo "After installing Docker, run this script again."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    echo ""
    echo "📋 How to start Docker:"
    echo "   • Windows/Mac: Open Docker Desktop application"
    echo "   • Linux: Run 'sudo systemctl start docker'"
    exit 1
fi

echo -e "${GREEN}✅ Docker is installed and running${NC}"
echo ""

# Create data directory if it doesn't exist
if [ ! -d "../data" ]; then
    echo -e "${YELLOW}📁 Creating data directory...${NC}"
    mkdir -p ../data
fi

# Build and run the application
echo -e "${YELLOW}🔨 Building AML Dashboard Docker image...${NC}"
docker-compose build

echo -e "${YELLOW}🚀 Starting AML Dashboard...${NC}"
docker-compose up -d

# Wait for the application to start
echo -e "${YELLOW}⏳ Waiting for application to start...${NC}"
sleep 10

# Check if the application is running
if curl -s http://localhost:8502 > /dev/null; then
    echo ""
    echo -e "${GREEN}🎉 AML Dashboard is now running!${NC}"
    echo ""
    echo -e "${BLUE}📋 Access Information:${NC}"
    echo "   • Local URL: http://localhost:8502"
    echo "   • Network URL: http://$(hostname -I | awk '{print $1}'):8502"
    echo ""
    echo -e "${BLUE}📁 Data Directory:${NC}"
    echo "   • Place your CSV files in the '../data' folder"
    echo "   • The application will automatically detect them"
    echo ""
    echo -e "${BLUE}🔧 Management Commands:${NC}"
    echo "   • Stop dashboard: ./stop-aml-dashboard.sh"
    echo "   • View logs: docker-compose logs -f"
    echo "   • Restart: docker-compose restart"
    echo ""
    echo -e "${GREEN}✅ Your AML Dashboard is ready to use!${NC}"
    echo ""
    echo -e "${YELLOW}💡 Tip: Open your web browser and go to http://localhost:8502${NC}"
else
    echo -e "${RED}❌ Application failed to start. Check logs with: docker-compose logs${NC}"
    exit 1
fi 