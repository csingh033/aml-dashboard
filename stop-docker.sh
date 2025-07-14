#!/bin/bash

# AML Dashboard - Docker Stopper
# This script stops the AML dashboard Docker container

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 AML Dashboard - Docker Stopper${NC}"
echo -e "${BLUE}===============================${NC}"
echo ""

# Change to docker directory and stop
cd docker
./stop-aml-dashboard.sh 