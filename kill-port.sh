#!/bin/bash

# Kill Port Script
# This script kills processes running on specified ports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔪 Kill Port Script${NC}"
echo -e "${BLUE}=================${NC}"
echo ""

# Function to kill process on a specific port
kill_port() {
    local port=$1
    echo -e "${YELLOW}🔍 Checking for processes on port $port...${NC}"
    
    # Check if any process is using the port
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}🔄 Killing processes on port $port...${NC}"
        lsof -ti:$port | xargs kill -9
        echo -e "${GREEN}✅ Successfully killed processes on port $port${NC}"
    else
        echo -e "${GREEN}✅ No processes found on port $port${NC}"
    fi
}

# Kill common ports
kill_port 8501
kill_port 8502
kill_port 8503

echo ""
echo -e "${GREEN}✅ Port cleanup completed!${NC}"
echo ""
echo -e "${BLUE}📋 Usage:${NC}"
echo "   • ./kill-port.sh - Kill processes on common ports"
echo "   • lsof -ti:PORT | xargs kill -9 - Kill specific port" 