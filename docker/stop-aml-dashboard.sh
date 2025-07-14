#!/bin/bash

# AML Dashboard - Stop Script
# This script stops the AML dashboard

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 AML Dashboard - Stop Script${NC}"
echo -e "${BLUE}============================${NC}"
echo ""

echo -e "${YELLOW}🛑 Stopping AML Dashboard...${NC}"
docker-compose down

echo -e "${GREEN}✅ AML Dashboard has been stopped successfully!${NC}"
echo ""
echo -e "${BLUE}📋 To start again, run: ./run-aml-dashboard.sh${NC}" 