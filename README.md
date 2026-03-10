<div align="center">

# VEXIS-CLI-1

</div>

<div align="center">

![VEXIS CLI Logo](https://img.shields.io/badge/VEXIS-CLI%201.0.0-blue?style=for-the-badge)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey?style=for-the-badge)

**AI-Powered Command Line Interface for Intelligent Automation**

[VEXIS-CLI-1](https://github.com/AInohogosya-team/VEXIS-CLI-1) is an AI agent derived from VEXIS-1.1 that performs tasks through command execution. It leverages large language models to intelligently interpret natural language commands and execute them through terminal operations, enabling automated workflow management and system administration.

[Quick Start](#quick-start) • [Documentation](#documentation) • [Models](#supported-ai-models) • [Configuration](#configuration) • [Contributing](#contributing)

</div>

## Key Features

### AI-Powered Intelligence
- **Natural Language Processing**: Execute commands using plain English descriptions
- **Context-Aware Execution**: Understands your workflow and adapts to your needs
- **Multi-Model Support**: Compatible with 80+ AI models from 12 major providers
- **Smart Verification**: Automatic task completion validation with confidence scoring

### Advanced Automation
- **Two-Phase Engine**: Planning and execution phases for reliable task completion
- **Cross-Platform Compatibility**: Works seamlessly on macOS, Linux, and Windows
- **GUI Automation**: Integrate terminal commands with graphical interface interactions
- **Screenshot Integration**: Visual context capture for enhanced understanding

### Developer Experience
- **Rich Terminal UI**: Beautiful, informative output with progress indicators
- **Flexible Configuration**: YAML-based settings with environment variable overrides
- **Extensible Architecture**: Plugin-ready design for custom integrations
- **Comprehensive Logging**: Structured logging for debugging and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.com/) installed and running (for local AI models)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AInohogosya-team/VEXIS-CLI-1.git
   cd VEXIS-CLI-1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama** (optional, for local models)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull a recommended model
   ollama pull llama3.2:latest
   ```

4. **Run VEXIS-CLI**
   ```bash
   python run.py
   ```

### Your First Command

```bash
# Start the interactive interface
python run.py

# Or use direct commands
vexis-cli "List all Python files in the current directory"
```

## Supported AI Models

VEXIS-CLI-1 supports **150+ models** from **20 major providers** through Ollama:

### Core Providers
- **Meta**: Llama 3.1/3.2 series (8B, 70B, 1B, 3B variants)
- **Google**: Gemma 2/3 series (1B-27B parameters, multimodal capabilities)
- **DeepSeek**: R1/V3/Coder series (8B-671B, reasoning and coding specialists)
- **Microsoft**: Phi-3/4 series (3.8B-14B, efficient small models)
- **Mistral**: Mistral/Large/Ministral series (7B-675B, European open-source leader)

### Advanced Providers
- **Alibaba (Qwen)**: Qwen 2.5/3 series (0.5B-235B, multilingual with 128K+ context)
- **IBM**: Granite/Code series (350M-34B, enterprise-grade models)
- **BigCode**: StarCoder 2 series (3B-15B, specialized for code generation)
- **Cohere**: Command R series (7B-35B, retrieval-augmented generation)
- **01.AI**: Yi series (1.5B-34B, bilingual models)

### Specialized Models
- **Vision-Language**: LLaVA, Moondream, Qwen3-VL (7B-235B)
- **Coding**: DeepSeek Coder, Qwen Coder, Granite Code, StarCoder 2
- **Agentic**: Hermes 3, Reflection, Devstral Small 2 (3B-405B)
- **Cloud-Only**: GPT-OSS, Gemini 3, GLM-5, MiniMax, Kimi (20B-744B)

### Cloud & Local Models
- **Local Models**: Run entirely on your machine with Ollama
- **Cloud Models**: Access high-performance models via API
- **Hybrid Mode**: Seamlessly switch between local and cloud models

<details>
<summary>Complete Model List</summary>

**Popular Local Models:**
- `llama3.2:latest` (3B) - Balanced performance with 128K context
- `qwen2.5:7b` - Multilingual capabilities with 128K context
- `mistral:7b` - Fast and efficient with 32K context
- `deepseek-r1:8b` - Advanced reasoning with 128K context
- `gemma2:9b` - High-performing with 8K context
- `phi3:mini` - Efficient small model with 4K context

**High-Performance Cloud Models:**
- `deepseek-v3:671b` - State-of-the-art MoE with 160K context
- `qwen3:235b` - Advanced MoE with 256K context
- `mistral-large-3:675b-cloud` - Multimodal enterprise model
- `gpt-oss:120b-cloud` - Frontier performance
- `gemini-3-flash-preview:cloud` - Built for speed

</details>

## Usage Examples

### Quick Start
```bash
# Start the interactive interface
python run.py

# Or use direct commands
vexis-cli "List all Python files in the current directory"
```

For detailed usage examples, see our [Detailed Guide](DETAILED_GUIDE.md).

## Documentation

- [📖 Detailed Guide](DETAILED_GUIDE.md) - Comprehensive usage examples and advanced features
- [🔧 Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [📚 API Reference](docs/API_REFERENCE.md)
- [🏗️ Architecture](docs/ARCHITECTURE.md)
- [⚙️ Configuration](docs/CONFIGURATION.md)
- [🤝 Contributing](docs/CONTRIBUTING.md)

## Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/AInohogosya-team/VEXIS-CLI-1/issues)
- **Discussions**: [Join the community discussion](https://github.com/AInohogosya-team/VEXIS-CLI-1/discussions)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: AInohogosya@proton.me
- X: [AInohogosya](https://twitter.com/AInohogosya)
- Home Page: https://ainohogosya.github.io/home-page/

---

<div align="center">

[Back to top](#vexis-cli-1)

Made with ❤️ by the [AInohogosya-team](https://github.com/AInohogosya-team)

</div>
