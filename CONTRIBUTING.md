# Contributing to XAI Robustness Benchmark

Thank you for your interest in contributing to the XAI Robustness Benchmark! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/xai-robustness-benchmark.git
   cd xai-robustness-benchmark
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 🛠️ Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/your-bugfix-name
```

### Making Changes

1. Make your changes in your feature branch
2. Add tests for new functionality
3. Ensure all tests pass: `pytest tests/`
4. Follow the code style guidelines (PEP 8)

### Committing Changes

Use clear and descriptive commit messages:

```bash
git add .
git commit -m "Add: New XAI method implementation for [Method Name]"
# or
git commit -m "Fix: Corruption implementation bug in [Corruption Type]"
# or
git commit -m "Docs: Update README with new examples"
```

### Submitting a Pull Request

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. Open a Pull Request on GitHub
3. Provide a clear description of your changes
4. Reference any related issues

## 📋 Contribution Types

### Adding a New XAI Method

1. Create a new evaluator in `methods/`:
   ```python
   # methods/your_method_evaluator.py
   from common.base_xai_evaluator import BaseXAIEvaluator
   
   class YourMethodEvaluator(BaseXAIEvaluator):
       def generate_explanation(self, image, target_class):
           # Implementation
           pass
   ```

2. Add configuration in `configs/experiment_config.yaml`
3. Add tests in `tests/test_your_method.py`
4. Update documentation

### Adding a New Corruption Type

1. Implement the corruption in `common/corruptions.py`:
   ```python
   def apply_your_corruption(image, severity):
       # Implementation
       pass
   ```

2. Add to corruption registry
3. Add tests
4. Update documentation

### Adding a New Metric

1. Implement the metric in `common/metrics.py`:
   ```python
   def compute_your_metric(original, corrupted):
       # Implementation
       pass
   ```

2. Add to metric computation pipeline
3. Add tests
4. Update documentation

## ✅ Code Quality Guidelines

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and concise

### Documentation

- Update README.md for user-facing changes
- Add docstrings with clear descriptions
- Include usage examples
- Comment complex logic

### Testing

- Write unit tests for new functionality
- Ensure existing tests pass
- Aim for high test coverage
- Include edge cases

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. A clear and descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Your environment (Python version, OS, dependencies)
5. Any relevant code snippets or error messages

## 💡 Suggesting Enhancements

When suggesting enhancements:

1. Use a clear and descriptive title
2. Provide a detailed description of the proposed feature
3. Explain why this enhancement would be useful
4. Include examples of how it would be used

## 📝 Code Review Process

1. All submissions require review before merging
2. Reviewers will check:
   - Code quality and style
   - Test coverage
   - Documentation completeness
   - Backward compatibility
3. Address review comments promptly
4. Be open to constructive feedback

## 🏆 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Acknowledged in release notes
- Credited in relevant documentation

## 📧 Contact

For questions about contributing:
- Open an issue with the "question" label
- Email: your.email@example.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to XAI Robustness Benchmark! 🎉

