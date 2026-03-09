# VEXIS-CLI-1

<div align="center">

![VEXIS CLI Logo](https://img.shields.io/badge/VEXIS-CLI%201.0-blue?style=for-the-badge)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey?style=for-the-badge)

**AI-Powered Command Line Interface for Intelligent Automation**

[VEXIS-CLI-1](https://github.com/AInohogosya-team/VEXIS-CLI-1) is a sophisticated AI agent that brings the power of large language models to your terminal, enabling intelligent task automation, natural language command processing, and seamless workflow integration.

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

VEXIS-CLI-1 supports **80+ models** from **12 major providers**:

### Major Providers
- **OpenAI**: GPT-4, GPT-3.5 series
- **Meta**: Llama 3.1/3.2/3.3 series
- **Google**: Gemma 2/3, Gemini models
- **Mistral**: Mistral Large/Small, Ministral
- **Anthropic**: Claude 3.5/3.7 series
- **DeepSeek**: DeepSeek R1, DeepSeek Coder

### Emerging Providers
- **Alibaba (Qwen)**: Qwen 2.5/3 series
- **Zhipu AI**: GLM-4/4.7/5 series
- **IBM**: Granite 3 series
- **01.AI**: Yi series
- **BigCode**: StarCoder series
- **Cohere**: Command R series

### Cloud & Local Models
- **Local Models**: Run entirely on your machine with Ollama
- **Cloud Models**: Access high-performance models via API
- **Hybrid Mode**: Seamlessly switch between local and cloud models

<details>
<summary>Complete Model List</summary>

**Popular Local Models:**
- `llama3.2:latest` (9B) - Balanced performance
- `qwen2.5:7b` - Multilingual capabilities
- `mistral:7b` - Fast and efficient
- `deepseek-coder` - Specialized for coding

**High-Performance Cloud Models:**
- `gpt-4o` - Advanced reasoning
- `claude-3.5-sonnet` - Complex task handling
- `gemini-3-flash-preview` - Speed and accuracy
- `deepseek-r1:671b-cloud` - Massive scale reasoning

</details>

## Configuration

### Basic Configuration

Edit `config.yaml` to customize your experience:

```yaml
api:
  preferred_provider: "ollama"
  local_endpoint: "http://localhost:11434"
  local_model: "llama3.2:latest"
  timeout: 120

verification:
  enabled: true
  confidence_threshold: 0.8
  max_verification_attempts: 3

engine:
  click_delay: 0.1
  typing_delay: 0.05
  screenshot_quality: 95
```

### Environment Variables

Override configuration with environment variables:

```bash
export AI_AGENT_LOCAL_ENDPOINT="http://localhost:11434"
export AI_AGENT_PREFERRED_PROVIDER="ollama"
export AI_AGENT_LOCAL_MODEL="llama3.2:latest"
```

### Model Selection

Switch between models dynamically:

```bash
# Use a specific model
vexis-cli --model deepseek-coder "Review this Python code"

# Use cloud model
vexis-cli --model gpt-4o "Analyze this system architecture"
```

## Usage Examples

### File Operations
```bash
# Natural language file management
vexis-cli "Find all Python files over 1MB and move them to the archive folder"

# Batch processing
vexis-cli "Convert all PNG images in this directory to WebP format"
```

### Code Development
```bash
# Code review
vexis-cli "Review this pull request and suggest improvements"

# Documentation
vexis-cli "Generate API documentation for the src/ai_agent module"
```

### System Administration
```bash
# System monitoring
vexis-cli "Check disk usage and alert if any partition is over 80% full"

# Log analysis
vexis-cli "Analyze the last 1000 lines of application.log for error patterns"
```

### Workflow Automation
```bash
# Multi-step tasks
vexis-cli "Set up a new Python project with virtual environment, install requirements, and initialize git"

# Scheduled tasks
vexis-cli "Create a cron job to backup the database every Sunday at 2 AM"
```

## Architecture

```
VEXIS-CLI-1 Architecture
├── 🧠 AI Agent Core
│   ├── Natural Language Processing
│   ├── Task Planning & Execution
│   └── Verification Engine
├── 🔌 External Integration
│   ├── Ollama Interface
│   ├── Cloud API Connectors
│   └── Platform Abstraction
├── 🎨 User Interface
│   ├── Rich Terminal Display
│   ├── Interactive Mode
│   └── Progress Indicators
└── 🛠️ Utilities
    ├── Configuration Management
    ├── Logging & Monitoring
    └── Error Handling
```

## Advanced Features

### Two-Phase Execution Engine

1. **Planning Phase**: Analyzes the request and creates an execution plan
2. **Execution Phase**: Carries out the plan with real-time monitoring
3. **Verification Phase**: Validates task completion and handles errors

### Smart Context Management

- **Session Memory**: Maintains context across multiple commands
- **File Awareness**: Understands your project structure
- **History Tracking**: Learns from your usage patterns

### Error Recovery

- **Automatic Retries**: Intelligent retry logic with exponential backoff
- **Fallback Strategies**: Switches to alternative approaches on failure
- **Detailed Logging**: Comprehensive error reporting for debugging

## Development

### Setting Up Development Environment

1. **Clone and install**
   ```bash
   git clone https://github.com/AInohogosya-team/VEXIS-CLI-1.git
   cd VEXIS-CLI-1
   pip install -e ".[dev]"
   ```

2. **Run tests**
   ```bash
   python -m pytest tests/
   ```

3. **Development mode**
   ```bash
   python run.py --debug
   ```

### Project Structure

```
src/ai_agent/
├── core_processing/          # AI processing logic
├── external_integration/     # External API integrations
├── platform_abstraction/     # Cross-platform compatibility
├── user_interface/          # CLI and TUI components
└── utils/                    # Utility functions
```

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Performance

### Benchmarks

| Task | Local Model (Llama 3.2) | Cloud Model (GPT-4o) |
|------|-------------------------|---------------------|
| Simple Commands | ~2s | ~1s |
| Code Generation | ~5s | ~2s |
| Complex Analysis | ~15s | ~5s |
| Multi-step Tasks | ~30s | ~10s |

### Resource Usage

- **Memory**: 2-8GB depending on model size
- **CPU**: Moderate usage during processing
- **Network**: Minimal for local models, variable for cloud models

## Troubleshooting

### Common Issues

<details>
<summary>Ollama Connection Failed</summary>

**Problem**: Cannot connect to Ollama server
**Solution**: 
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve

# Check configuration
cat ~/.ollama/config
```
</details>

<details>
<summary>Model Not Found</summary>

**Problem**: Model not available in Ollama
**Solution**:
```bash
# Pull the model
ollama pull llama3.2:latest

# List available models
ollama list
```
</details>

<details>
<summary>Permission Denied</summary>

**Problem**: Cannot access certain files or directories
**Solution**:
```bash
# Check permissions
ls -la /path/to/file

# Fix permissions (if safe)
chmod 644 /path/to/file
```
</details>

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export AI_AGENT_DEBUG=true
python run.py --debug
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Configuration Guide](docs/CONFIGURATION.md)
- [Contributing Guide](docs/CONTRIBUTING.md)

## Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/AInohogosya-team/VEXIS-CLI-1/issues)
- **Discussions**: [Join the community discussion](https://github.com/AInohogosya-team/VEXIS-CLI-1/discussions)
- **Wiki**: [Community-maintained documentation](https://github.com/AInohogosya-team/VEXIS-CLI-1/wiki)

## Roadmap

### Version 1.1 (Q2 2026)
- [ ] Enhanced GUI automation capabilities
- [ ] Additional cloud provider integrations
- [ ] Performance optimizations
- [ ] Plugin system

### Version 1.2 (Q3 2026)
- [ ] Multi-modal AI support (vision, audio)
- [ ] Advanced workflow orchestration
- [ ] Team collaboration features
- [ ] Mobile app companion

### Version 2.0 (Q4 2026)
- [ ] Distributed processing
- [ ] Advanced security features
- [ ] Enterprise integrations
- [ ] Custom model training support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ollama](https://ollama.com/) for making local AI accessible
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- The open-source community for inspiration and contributions
- All our beta testers and early adopters

## Support

- 📧 Email: support@vexis-cli.com
- 💬 Discord: [Join our Discord server](https://discord.gg/vexis-cli)
- 🐦 Twitter: [@VEXIS_CLI](https://twitter.com/VEXIS_CLI)

---

<div align="center">

[Back to top](#vexis-cli-1)

Made with ❤️ by the [AInohogosya-team](https://github.com/AInohogosya-team)

</div>
