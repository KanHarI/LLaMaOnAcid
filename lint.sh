#!/bin/bash

set -e

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print styled messages
print_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Define the Python source directories to check
PYTHON_DIRS="llama_on_acid example.py run_experiment.py"

# Print script header
echo -e "${YELLOW}===============================================${NC}"
echo -e "${YELLOW}     Running code quality checks on codebase   ${NC}"
echo -e "${YELLOW}===============================================${NC}"

# Run Black (code formatter)
print_header "Running Black code formatter"
if black --check $PYTHON_DIRS; then
    print_success "Black: Code is well formatted."
else
    echo -e "${YELLOW}Black: Reformatting code...${NC}"
    black $PYTHON_DIRS
    print_success "Black: Code has been formatted."
fi

# Run isort (import sorter)
print_header "Running isort import sorter"
if isort --check $PYTHON_DIRS; then
    print_success "isort: Imports are well organized."
else
    echo -e "${YELLOW}isort: Reorganizing imports...${NC}"
    isort $PYTHON_DIRS
    print_success "isort: Imports have been reorganized."
fi

# Run mypy (type checker)
print_header "Running mypy type checker"
if mypy $PYTHON_DIRS; then
    print_success "mypy: No type errors found."
else
    print_error "mypy: Type errors found. Please fix them before committing."
fi

# Run flake8 (linter)
print_header "Running flake8 linter"
if flake8 $PYTHON_DIRS; then
    print_success "flake8: No linting errors found."
else
    print_error "flake8: Linting errors found. Please fix them before committing."
fi

echo -e "\n${GREEN}===============================================${NC}"
echo -e "${GREEN}     All code quality checks passed!           ${NC}"
echo -e "${GREEN}===============================================${NC}" 