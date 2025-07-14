#!/bin/bash

# AML Dashboard - Docker Runner
# This script runs the AML dashboard using Docker

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AML Dashboard - Docker Runner${NC}"
echo -e "${BLUE}=============================${NC}"
echo ""

# Change to docker directory and run
cd docker
./run-aml-dashboard.sh 