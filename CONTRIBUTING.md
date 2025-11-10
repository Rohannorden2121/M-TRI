# Contributing to M-TRI

### Prerequisites

- Python 3.8+
- Git
- Docker (for containerized development)
- Basic understanding of machine learning and environmental data

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

3. **Test changes**
   ```bash
   # Run tests
   pytest tests/ -v --cov=src
   
   # Run linting
   flake8 src/ tests/
   black --check src/ tests/
   
   # Type checking
   mypy src/
   ```
Thank you for your contributions to M-TRI.
