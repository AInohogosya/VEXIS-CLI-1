
<div align="center">

![Betelgeuse](image.png)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange?style=flat-square)]()

**AI agent that automates command-line tasks**

</div>

---

## About

**VEXIS-CLI-1.1** is an AI command-line agent that processes natural language instructions to execute terminal operations. Supports multiple AI providers including Ollama, Google Gemini, OpenAI, Anthropic, xAI, Meta, Mistral AI, Microsoft Azure, Amazon Bedrock, Cohere, DeepSeek, Groq, and Together AI.

### AI Providers

**A provider that supports both cloud and local environments:**
- **Ollama**: You can use Ollama's local models or cloud models. It is recommended to sign in to Ollama in advance.

**Cloud Providers (API Key Required):**
- **Groq**: Fast inference with Llama/Mixtral models
- **Google Gemini**: Enterprise-grade cloud models  
- **OpenAI**: GPT models with advanced capabilities
- **Anthropic**: Claude models with strong reasoning
- **xAI**: Grok models for real-time knowledge
- **Meta**: Llama models via Meta API
- **Mistral AI**: Advanced multilingual models
- **Microsoft Azure**: GPT models via Azure
- **Amazon Bedrock**: Titan/Nova models via AWS
- **Cohere**: Command models for enterprise
- **DeepSeek**: Advanced reasoning models
- **Together AI**: Open-source model hosting

> **🎯 Now Functional**: All 13 providers are now integrated and working through the unified API system!

> **💡 AInohogosya-Team Recommendations**: For best performance and reliability, we recommend **Groq** (fast inference), **Google Gemini** (enterprise-grade), **OpenAI** (advanced capabilities), and **Ollama** (privacy-first local).

> **⚠️ Provider Compatibility**: While we support 13+ AI providers, using other providers beyond our recommendations may occasionally encounter errors or bugs due to API differences or model-specific behaviors.

> **⚠️ Important Note**: Some Ollama models may experience compatibility issues or errors. If you encounter problems with specific models, try alternatives like `gemma3:4b`, `qwen2.5:3b`, or `deepseek-r1:7b`.

> **Note**: Experimental project. Use with curiosity!

---

## Features

- Natural language to CLI conversion
- Command execution with intelligent error handling
- File operations and workflow automation
- Enhanced Ollama error handling with user-friendly guidance
- One-liner execution: `python3 run.py "do something"`

---

## Installation

```bash
git clone https://github.com/AInohogosya-team/VEXIS-CLI-1.1.git
cd VEXIS-CLI-1.1
python3 run.py "list files"  # Dependencies handled automatically
```

### Requirements
- Python 3.9+
- API keys for cloud providers (Ollama works locally)
- Optional: Ollama account for cloud models (`ollama signin`)

> 📖 **Need detailed guidance?** Check out our [Detailed User Guide](./DETAILED_GUIDE.md) for comprehensive setup instructions.

---

## Usage

### Provider Selection

When you run VEXIS-CLI, you'll first see a provider selection screen where you can choose your preferred AI provider:

![Provider Selection Screen](Choose_model.png)

### Commands

```bash
# Basic commands
python3 run.py "list files in current directory"
python3 run.py "create hello.txt with content 'Hello World'"
python3 run.py "show system information"

# Options
python3 run.py "instruction" --debug     # Verbose logging
python3 run.py "instruction" --no-prompt # Skip provider selection
```

---

## Configuration

Edit `config.yaml`:

```yaml
api:
  preferred_provider: "ollama"  # "ollama", "groq", "google", "openai", "anthropic", "xai", "meta", "mistral", "azure", "amazon", "cohere", "deepseek", "together"
  local_endpoint: "http://localhost:11434"
  local_model: "gemma3:4b"  # Recommended stable model
  timeout: 120
  max_retries: 3
```

**Model Recommendations:**
- **Stable**: `gemma3:4b`, `qwen2.5:3b`, `deepseek-r1:7b`
- **Latest**: `gemini-3.1-pro`, `gpt-5.4`, `claude-opus-4.6`
- **Cloud**: Google Gemini 3.1 Pro (most reliable)

> 📖 **For detailed configuration options**, see our [Configuration Guide](./docs/CONFIGURATION.md) and [Detailed User Guide](./DETAILED_GUIDE.md).

---

## Error Handling

Comprehensive Ollama error guidance:
- **Permission Errors**: macOS Full Disk Access, Linux permissions, Windows admin
- **Model Errors**: Available models list and alternative suggestions
- **Connection Issues**: Service restart and port checking
- **Installation Problems**: Platform-specific instructions

> 📖 **Need help with issues?** Check our [Troubleshooting Guide](./docs/TROUBLESHOOTING.md) and [Detailed User Guide](./DETAILED_GUIDE.md#troubleshooting).

---

## Architecture

Two-phase execution engine:
1. **Command Planning**: Natural language analysis
2. **Terminal Execution**: Implementation with error recovery

### Core Components
- `TwoPhaseEngine` - Orchestration
- `ModelRunner` - AI provider abstraction
- `CommandParser` - Natural language processing
- `TaskVerifier` - Validation and error handling

---

<div align="center">

**VEXIS-CLI-1.1 - Intelligent command-line automation**

</div>
