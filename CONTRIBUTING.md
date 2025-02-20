# Contributing to M-TRI

Thank you for your interest in contributing to the Microbial Toxin-Risk Index (M-TRI) project! This document provides guidelines for contributing to the codebase.

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker (for containerized development)
- Basic understanding of machine learning and environmental data

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/m-tri.git
   cd m-tri
   git remote add upstream https://github.com/original-username/m-tri.git
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature development
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v --cov=src
   
   # Run linting
   flake8 src/ tests/
   black --check src/ tests/
   
   # Type checking
   mypy src/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python Code Standards

- **Formatting**: Use `black` for consistent formatting
- **Linting**: Follow `flake8` guidelines
- **Type Hints**: Use type annotations for function signatures
- **Docstrings**: Use Google-style docstrings
- **Line Length**: Maximum 88 characters (black default)

### Example Code Style

```python
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np


class FeatureEngineer:
    """Feature engineering pipeline for environmental data.
    
    This class handles transformation of raw environmental measurements
    into model-ready features for HAB prediction.
    
    Args:
        temporal_window: Days to include in rolling calculations
        spatial_buffer: Meters for spatial feature aggregation
        
    Example:
        >>> engineer = FeatureEngineer(temporal_window=30)
        >>> features = engineer.create_features(water_df, satellite_df)
    """
    
    def __init__(
        self, 
        temporal_window: int = 30,
        spatial_buffer: float = 1000.0
    ) -> None:
        self.temporal_window = temporal_window
        self.spatial_buffer = spatial_buffer
        
    def create_features(
        self,
        water_quality: pd.DataFrame,
        satellite_data: pd.DataFrame,
        weather: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create model features from raw environmental data.
        
        Args:
            water_quality: Water chemistry measurements
            satellite_data: Remote sensing indices
            weather: Optional meteorological data
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            ValueError: If required columns are missing
        """
        # Implementation here
        pass
```

### Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
- `feat(api): add batch prediction endpoint`
- `fix(ingestion): handle missing USGS data gracefully`
- `docs(readme): update installation instructions`

## Testing Guidelines

### Test Structure

```
tests/
├── unit/               # Unit tests for individual functions
├── integration/        # Integration tests for components
├── fixtures/          # Test data and fixtures
└── conftest.py       # Pytest configuration
```

### Writing Tests

```python
import pytest
import pandas as pd
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for feature engineering pipeline."""
    
    @pytest.fixture
    def sample_water_data(self):
        """Sample water quality data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'temperature': np.random.normal(20, 5, 100),
            'ph': np.random.normal(7.5, 0.5, 100),
            'chlorophyll_a': np.random.exponential(10, 100)
        })
    
    def test_feature_creation(self, sample_water_data):
        """Test basic feature creation functionality."""
        engineer = FeatureEngineer(temporal_window=7)
        features = engineer.create_features(sample_water_data, None)
        
        assert not features.empty
        assert 'temperature_7d_mean' in features.columns
        assert features.shape[0] > 0
    
    def test_missing_data_handling(self):
        """Test handling of missing input data."""
        engineer = FeatureEngineer()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.create_features(pd.DataFrame(), None)
```

### Test Coverage

- Maintain >80% test coverage
- Test both happy path and error conditions
- Include edge cases and boundary conditions
- Mock external API calls and file I/O

## Documentation

### Code Documentation

- **Docstrings**: All public functions and classes must have docstrings
- **Type Hints**: Use type annotations consistently
- **Comments**: Explain complex logic and business rules
- **Examples**: Include usage examples in docstrings

### Project Documentation

- Update `README.md` for user-facing changes
- Update `CHANGELOG.md` for all releases
- Add API documentation for new endpoints
- Include configuration documentation

## Bug Reports

### Before Submitting

1. Check existing issues for duplicates
2. Verify the bug with latest version
3. Create minimal reproduction case

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., macOS 12.0]
- Python: [e.g., 3.9.7]
- M-TRI version: [e.g., 1.0.0]

## Additional Context
Any other relevant information
```

## Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Implementation
High-level approach (optional)

## Alternatives Considered
Other solutions you've considered

## Additional Context
Any other relevant information
```

## Code Review Process

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Style**: Does it follow project conventions?
- **Documentation**: Is it properly documented?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security considerations?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact is acceptable
- [ ] Security implications considered

## Release Process

### Version Numbers

Follow semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. Update version numbers
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create release tag
6. Deploy to production
7. Announce release

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different opinions and experiences

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: Direct contact for security issues

## Getting Help

### Resources

- [Project Documentation](https://mtri-docs.readthedocs.io)
- [API Reference](https://api.mtri.example.com/docs)
- [GitHub Issues](https://github.com/username/m-tri/issues)
- [GitHub Discussions](https://github.com/username/m-tri/discussions)

### Asking Questions

1. Search existing issues and discussions
2. Provide clear, specific questions
3. Include relevant context and examples
4. Be patient and respectful

---

Thank you for contributing to M-TRI! Your efforts help protect public health and environmental resources.