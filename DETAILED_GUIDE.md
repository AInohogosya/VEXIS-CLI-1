# VEXIS-CLI-1.2 Detailed User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [AI Provider Setup](#ai-provider-setup)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Getting Started

### What is VEXIS-CLI-1.2?

VEXIS-CLI-1.2 is an intelligent command-line agent that transforms natural language instructions into executable terminal commands. It supports multiple AI providers and offers both local (privacy-first) and cloud-based options.

### Key Features

- **Natural Language Processing**: Convert plain English commands to terminal operations
- **Multi-Provider Support**: 13+ AI providers including local Ollama models
- **Error Handling**: Intelligent error recovery and user guidance
- **One-Liner Execution**: Simple `python3 run.py "your command"` syntax
- **Cross-Platform**: Works on macOS, Linux, and Windows

---

## Installation & Setup

### Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/vexis-project/VEXIS-CLI-1.2.git
cd VEXIS-CLI-1.2

# Run your first command (dependencies auto-installed)
python3 run.py "list files in current directory"
```

### Manual Installation

```bash
# 1. Clone the repository
git clone https://github.com/vexis-project/VEXIS-CLI-1.2.git
cd VEXIS-CLI-1.2

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the agent
python3 run.py "your instruction here"
```

### System Requirements

- **Python**: 3.9 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space
- **Network**: Internet connection for cloud providers

---

## Configuration

### Basic Configuration

Edit `config.yaml` to customize your setup:

```yaml
api:
  preferred_provider: "ollama"  # Your preferred AI provider
  local_endpoint: "http://localhost:11434"  # Ollama endpoint
  local_model: "gemma3:4b"  # Default local model
  timeout: 120  # Request timeout in seconds
  max_retries: 3  # Maximum retry attempts

verification:
  enabled: true  # Enable task verification
  confidence_threshold: 0.8  # Verification confidence threshold
  max_verification_attempts: 3  # Max verification tries

engine:
  command_timeout: 30  # Command execution timeout
  task_timeout: 300  # Overall task timeout
  max_task_retries: 3  # Maximum task retries
```

### Environment Variables

You can override configuration using environment variables:

```bash
export AI_AGENT_LOCAL_ENDPOINT="http://localhost:11434"
export AI_AGENT_PREFERRED_PROVIDER="ollama"
export AI_AGENT_LOCAL_MODEL="llama3.2:latest"
```

---

## Usage Examples

### Basic Commands

```bash
# File operations
python3 run.py "create a file called hello.txt with content 'Hello World'"
python3 run.py "list all Python files in the current directory"
python3 run.py "copy hello.txt to backup_hello.txt"

# System information
python3 run.py "show system information"
python3 run.py "check disk usage"
python3 run.py "list running processes"

# Development tasks
python3 run.py "create a new Python project structure"
python3 run.py "install requirements from requirements.txt"
python3 run.py "run tests for the current project"
```

### Advanced Commands

```bash
# With options
python3 run.py "deploy the application" --debug
python3 run.py "setup development environment" --no-prompt

# Complex workflows
python3 run.py "create a backup of all configuration files"
python3 run.py "setup a Python development environment with Django"
```

---

## AI Provider Setup

### Local Providers

#### Ollama Setup

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Start Ollama Service**:
   ```bash
   ollama serve
   ```

3. **Download Models**:
   ```bash
   ollama pull gemma3:4b
   ollama pull qwen2.5:3b
   ollama pull deepseek-r1:7b
   ```

4. **Configure VEXIS-CLI**:
   ```yaml
   api:
     preferred_provider: "ollama"
     local_model: "gemma3:4b"
   ```

### Cloud Providers

#### Google Gemini

1. **Get API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key

2. **Configure**:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. **Update config**:
   ```yaml
   api:
     preferred_provider: "google"
   ```

#### OpenAI

1. **Get API Key**:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key

2. **Configure**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

#### Anthropic

1. **Get API Key**:
   - Visit [Anthropic Console](https://console.anthropic.com/)
   - Create a new API key

2. **Configure**:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

#### Groq

1. **Get API Key**:
   - Visit [Groq Console](https://console.groq.com/)
   - Create a new API key

2. **Configure**:
   ```bash
   export GROQ_API_KEY="your-api-key-here"
   ```

#### Additional Providers

Similar setup process for:
- **xAI**: `XAI_API_KEY` (Grok models)
- **Meta**: `META_API_KEY` (Llama models)
- **Mistral AI**: `MISTRAL_API_KEY` (Mistral models)
- **Microsoft Azure**: `AZURE_API_KEY` (GPT via Azure)
- **Amazon Bedrock**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- **Cohere**: `COHERE_API_KEY` (Command models)
- **DeepSeek**: `DEEPSEEK_API_KEY` (Reasoning models)
- **Together AI**: `TOGETHER_API_KEY` (Open-source hosting)

---

## Advanced Features

### Two-Phase Execution Engine

VEXIS-CLI-1.2 uses a sophisticated two-phase execution:

1. **Command Planning Phase**:
   - Natural language understanding
   - Task decomposition
   - Command generation

2. **Terminal Execution Phase**:
   - Command execution
   - Error handling
   - Result verification

### Task Verification

The system includes intelligent task verification:

```yaml
verification:
  enabled: true
  confidence_threshold: 0.8
  max_verification_attempts: 3
  auto_regenerate: true
```

### Error Recovery

Automatic error recovery mechanisms:
- **Retry Logic**: Automatic retries for transient failures
- **Fallback Models**: Switch to alternative models on failure
- **User Guidance**: Helpful error messages and suggestions

---

## Troubleshooting

### Common Issues

#### Ollama Connection Issues

```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve

# Check connection
curl http://localhost:11434/api/tags
```

#### Permission Errors (macOS)

1. **Full Disk Access**:
   - Go to System Preferences > Security & Privacy > Privacy
   - Add Terminal to Full Disk Access

2. **Permission Fix**:
   ```bash
   chmod +x run.py
   ```

#### Model Compatibility Issues

Some models may have compatibility issues. Recommended alternatives:
- `gemma3:4b` - Most stable
- `qwen2.5:3b` - Lightweight and reliable
- `deepseek-r1:7b` - Advanced reasoning

#### Dependency Issues

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Update pip
pip install --upgrade pip

# Clean install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Debug Mode

Enable verbose logging:

```bash
python3 run.py "your command" --debug
```

### Getting Help

1. Check the [Troubleshooting Guide](./docs/TROUBLESHOOTING.md)
2. Review [Error Handling Documentation](./docs/ERROR_HANDLING.md)
3. Check GitHub Issues
4. Join community discussions

---

## Best Practices

### Performance Optimization

1. **Choose the Right Model**:
   - Local: `gemma3:4b` for balance
   - Cloud: Google Gemini for reliability
   - Fast: Groq for speed

2. **Optimize Commands**:
   - Be specific in your instructions
   - Break complex tasks into smaller steps
   - Use appropriate model for task complexity

### Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Local Models**: Use Ollama for sensitive data
3. **Network**: Use secure connections for cloud providers

### Productivity Tips

1. **Aliases**: Create shell aliases for common tasks
2. **Scripts**: Build reusable command scripts
3. **Configuration**: Fine-tune settings for your workflow

---

## FAQ

### Q: Which provider should I use?
**A**: For privacy: Ollama. For speed: Groq. For reliability: Google Gemini.

### Q: How do I handle large tasks?
**A**: Break them into smaller, specific commands for better results.

### Q: Can I use multiple providers?
**A**: Yes, configure fallback providers in your config file.

### Q: Is my data secure?
**A**: Local Ollama models keep data on your machine. Cloud providers send data to their servers.

### Q: How do I update VEXIS-CLI?
**A**: `git pull origin main` and reinstall dependencies if needed.

---

## Support & Community

- **Documentation**: [Full docs](./docs/)
- **Issues**: [GitHub Issues](https://github.com/vexis-project/VEXIS-CLI-1.2/issues)
- **Updates**: Check the repository regularly for updates

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-12  
**Compatibility**: Python 3.9+
