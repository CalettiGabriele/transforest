# TransForest 🌱

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

**TransForest** is a high-performance ensemble learning library that boosts the reliability, speed, and quality of Large Language Model (LLM) systems through advanced ensemble methods implemented in Rust with Python bindings.

## 🚀 Key Features

- **🎯 Enhanced Stability**: Reduce response variability through ensemble consensus
- **⚡ Superior Performance**: Rust-powered implementations up to 5x faster than pure Python
- **💰 Cost Optimization**: Use smaller, cheaper models while maintaining accuracy
- **🔧 Easy Integration**: Simple Python decorators for existing LLM functions
- **🤖 AI-Enhanced Fusion**: Intelligent response ranking and fusion using LLMs
- **📊 Multiple Algorithms**: MBR, Majority Voting, and AI Blender methods

## 📦 Installation

```bash
pip install transforest
```

### Development Installation

```bash
git clone https://github.com/yourusername/transforest.git
cd transforest
pip install -e .
```

## 🔧 Quick Start

### Basic Usage

```python
import transforest as tf

@tf.majority_voting(num_calls=5)
def your_llm_function(prompt):
    # Your LLM call implementation
    return llm_response

result = your_llm_function("What is machine learning?")
print(result['selected_response'])
```

## 🛠️ Available Methods

| Method | Decorator | Description | Best For |
|--------|-----------|-------------|----------|
| **Minimum Bayes Risk (MBR)** | `@tf.minimum_bayes_risk(num_calls=N)` | Selects response with minimum average distance to all others | Maximum stability and consistency |
| **Majority Voting** | `@tf.majority_voting(num_calls=N)` | Clusters similar responses and selects from largest cluster | Consensus-based reliability |
| **AI Blender** | `@tf.blender(num_calls=N, inference_config=config)` | Uses LLM to intelligently rank and fuse responses | Highest quality through AI enhancement |

## 📊 Performance Benefits

Based on our benchmarks:

- **Speed**: 2-5x faster than pure Python implementations
- **Memory**: 60% more memory efficient
- **Stability**: 40% improvement in response consistency
- **Quality**: Enhanced response quality through intelligent fusion

## 🔍 Use Cases

- **Production LLM Applications**: Improve reliability and reduce variability
- **Cost Optimization**: Use smaller models with ensemble methods for better results
- **Research & Development**: Experiment with different ensemble strategies
- **High-Volume Systems**: Scale efficiently with Rust-powered performance


## 📚 Documentation

- [Quick Start Guide](notebooks/quick_start_guide.ipynb)
- [Why Ensemble Methods?](notebooks/why_ensemble.ipynb)
- [Performance Comparison](notebooks/why_transforest.ipynb)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**TransForest** - Making LLM ensemble methods fast, reliable, and accessible! 🌱✨
