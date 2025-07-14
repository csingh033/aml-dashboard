# 🤝 Contributing to AML Dashboard

Thank you for your interest in contributing to the AML Dashboard! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker Desktop
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/aml-dashboard.git
   cd aml-dashboard
   ```

## 🛠️ Development Setup

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Docker Development

```bash
# Build the Docker image
docker build -t aml-dashboard-dev .

# Run the container
docker run -p 8501:8501 aml-dashboard-dev

# Or use docker-compose
docker-compose up
```

## 📝 Code Style

### Python Style Guide

We follow PEP 8 style guidelines. Use these tools to maintain code quality:

```bash
# Install development dependencies
pip install black flake8 isort

# Format code
black .

# Sort imports
isort .

# Check code style
flake8 .
```

### File Naming

- Use snake_case for Python files and functions
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Documentation

- Add docstrings to all functions and classes
- Update README.md for new features
- Include examples in docstrings

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

### Test Structure

```
tests/
├── test_anomaly_detection.py
├── test_network_analysis.py
├── test_llm_investigator.py
└── test_data_processing.py
```

### Writing Tests

- Write tests for new features
- Ensure test coverage > 80%
- Use descriptive test names
- Mock external dependencies

## 🔄 Pull Request Process

### Before Submitting

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update requirements.txt if new dependencies

2. **Run Tests**
   ```bash
   pytest
   docker build -t aml-dashboard-test .
   ```

3. **Check Code Style**
   ```bash
   black .
   flake8 .
   ```

### Creating a Pull Request

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write your code
   - Add tests
   - Update documentation

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new anomaly detection algorithm"
   ```

4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill out the PR template

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Docker build succeeds
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
```

## 🐛 Reporting Issues

### Before Reporting

1. Check existing issues
2. Search closed issues
3. Try the latest version

### Issue Template

```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Docker Version: [e.g., 20.10.8]
- Browser: [e.g., Chrome 96]

## Additional Information
Screenshots, logs, etc.
```

## 📊 Development Guidelines

### Feature Development

1. **Plan Your Feature**
   - Create an issue describing the feature
   - Discuss implementation approach
   - Get feedback from maintainers

2. **Implementation**
   - Follow the existing code structure
   - Add appropriate tests
   - Update documentation

3. **Testing**
   - Unit tests for new functions
   - Integration tests for new features
   - Manual testing in different environments

### Code Review Process

1. **Automated Checks**
   - GitHub Actions run tests
   - Code style checks
   - Docker build verification

2. **Manual Review**
   - At least one maintainer reviews
   - Check for security issues
   - Verify documentation updates

3. **Approval Process**
   - All tests must pass
   - Code review approved
   - Documentation updated

## 🔒 Security

### Security Guidelines

- Never commit sensitive data (API keys, passwords)
- Use environment variables for configuration
- Validate all user inputs
- Follow secure coding practices

### Reporting Security Issues

For security issues, please email security@yourcompany.com instead of creating a public issue.

## 📚 Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [NetworkX Documentation](https://networkx.org/)

### Tools
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: For security issues

### Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

---

**Thank you for contributing to AML Dashboard! 🎉**

Your contributions help make this tool better for everyone in the AML community. 