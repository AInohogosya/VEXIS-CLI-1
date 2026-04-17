# System Architecture

## Overview

VEXIS-CLI-2 is built on a **5-Phase Pipeline Architecture** that processes natural language instructions through intelligent command generation, execution, evaluation, and summarization with robust error handling and recovery mechanisms.

## Core Architecture

### 5-Phase Pipeline Execution Engine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VEXIS-CLI-2 5-Phase System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 1: Command Suggestion                                           │ │
│  │ ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │ │
│  │ │ User Prompt  │───▶│ AI Analysis  │───▶│ Command Strategy │       │ │
│  │ └──────────────┘    └──────────────┘    └──────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 2: Command Extraction                                           │ │
│  │ ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │ │
│  │ │ Parse Plan   │───▶│ Extract Cmds │───▶│ Validate Syntax  │       │ │
│  │ └──────────────┘    └──────────────┘    └──────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 3: Command Execution                                          │ │
│  │ ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │ │
│  │ │ Terminal     │───▶│ Execute      │───▶│ Capture Output   │       │ │
│  │ │ Injection    │    │ Batch        │    │ & Logs           │       │ │
│  │ └──────────────┘    └──────────────┘    └──────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 4: Log Evaluation                                               │ │
│  │ ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │ │
│  │ │ Analyze      │───▶│ Detect       │───▶│ Decide: Continue │       │ │
│  │ │ Output       │    │ Errors       │    │ or Complete      │       │ │
│  │ └──────────────┘    └──────────────┘    └──────────────────┘       │ │
│  │                                                                       │ │
│  │         ┌───────────────────────────────────────────┐                  │ │
│  │         │ Contains "failure"?                       │                  │ │
│  │         │ Yes: Loop to Phase 2  │  No: Proceed to Phase 5            │ │
│  │         └───────────────────────────────────────────┘                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ Phase 5: Summary Generation                                           │ │
│  │ ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │ │
│  │ │ Full Log     │───▶│ AI Summary   │───▶│ User Report      │       │ │
│  │ │ History      │    │ Generation   │    │ Display          │       │ │
│  │ └──────────────┘    └──────────────┘    └──────────────────┘       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  AI Provider Layer (16+ Providers)                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Ollama  │ │ Google  │ │ OpenAI  │ │ Anthro  │ │ Groq    │ │ xAI     │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Meta    │ │ Mistral │ │ Azure   │ │ AWS     │ │ Cohere  │ │DeepSeek │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                          │
│  │Together │ │ MiniMax │ │ ZhipuAI │ │OpenRoutr│                          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. FivePhaseEngine (`src/ai_agent/core_processing/five_phase_engine.py`)

**Responsibilities:**
- Orchestrates the complete 5-phase pipeline execution
- Phase 1: Command Suggestion - AI analyzes intent and suggests approach
- Phase 2: Command Extraction - Precise command isolation and validation
- Phase 3: Command Execution - Safe terminal execution with monitoring
- Phase 4: Log Evaluation - Intelligent error analysis and retry logic
- Phase 5: Summary Generation - Comprehensive result reporting
- Manages Phase 2-4 iteration loop for error recovery
- Handles phase transitions and state management via `PipelineContext`
- Implements intelligent error recovery with max 10 iterations

**Key Methods:**
```python
def execute_instruction(self, user_prompt: str) -> PipelineContext
def _run_phase1(self, context: PipelineContext) -> bool  # Command Suggestion
def _run_phase2(self, context: PipelineContext) -> bool  # Command Extraction
def _run_phase3(self, context: PipelineContext) -> bool  # Command Execution
def _run_phase4(self, context: PipelineContext) -> bool  # Log Evaluation
def _run_phase5(self, context: PipelineContext) -> bool  # Summary Generation
```

### 2. ModelRunner (`src/ai_agent/external_integration/model_runner.py`)

**Responsibilities:**
- Unified AI provider abstraction
- Handles both Ollama and Google Gemini providers
- Manages API communication and error handling
- Provides consistent response formatting

**Provider Support:**
- **Ollama**: Local models with privacy-first design
- **Google Gemini**: Cloud-based with enterprise reliability

**Key Features:**
- Automatic provider selection
- Fallback mechanisms
- Timeout and retry handling
- Response parsing and validation

### 3. CommandParser (`src/ai_agent/core_processing/command_parser.py`)

**Responsibilities:**
- Natural language to CLI command conversion
- Command validation and sanitization
- Parameter extraction and type checking
- Security constraint enforcement

**Supported Operations:**
- File system operations (create, move, delete, copy)
- Process management (start, stop, monitor)
- System information gathering
- Network operations
- Package management

### 4. TaskVerifier (`src/ai_agent/core_processing/task_verifier.py`)

**Responsibilities:**
- Command validation and safety checks
- Execution result verification
- Error detection and classification
- Success criteria evaluation

**Validation Layers:**
1. **Syntax Validation**: Command structure and syntax
2. **Security Validation**: Permission and access checks
3. **Logic Validation**: Command sequence and dependencies
4. **Result Validation**: Expected outcome verification

## AI Provider Integration

### Ollama Integration

**Architecture:**
```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│ VEXIS-CLI-2       │    HTTP/REST    │   Ollama        │
│ Model Runner    │                │   Local Server  │
└─────────────────┘                └─────────────────┘
        │                                   │
        │                                   ▼
        │                          ┌─────────────────┐
        │                          │ Local Models    │
        │                          │ • Gemini 3 Flash│
        │                          │ • Open Source   │
        │                          └─────────────────┘
```

**Features:**
- Local model execution for privacy
- Support for Gemini 3 Flash and open-source models
- Enhanced error handling with user guidance
- Automatic model management

### Google Gemini Integration

**Architecture:**
```
┌─────────────────┐    HTTPS/API    ┌─────────────────┐
│ VEXIS-CLI-2       │    HTTPS/API    │ Google Cloud    │
│ Model Runner    │                │   Gemini API    │
└─────────────────┘                └─────────────────┘
        │                                   │
        │                                   ▼
        │                          ┌─────────────────┐
        │                          │ Gemini Models   │
        │                          │ • Gemini 3      │
        │                          │ • Enterprise    │
        │                          └─────────────────┘
```

**Features:**
- Cloud-based reliability and scaling
- Enterprise-grade performance
- Automatic API key management
- Enhanced accuracy and capabilities

## Error Handling Architecture

### Error Handler Chain

```
┌─────────────────┐    Error Event    ┌─────────────────┐
│ Command Executor│◄─────────────────│  Error Handler  │
└─────────────────┘                  └─────────────────┘
        │                                   │
        │                                   ▼
        │                          ┌─────────────────┐
        │                          │ Error Classifier│
        │                          │ • Type          │
        │                          │ • Severity      │
        │                          │ • Recovery      │
        │                          └─────────────────┘
        │                                   │
        │                                   ▼
        │                          ┌─────────────────┐
        │                          │ Recovery Engine │
        │                          │ • Retry Logic   │
        │                          │ • Fallback      │
        │                          │ • User Guidance │
        │                          └─────────────────┘
```

### Error Categories

1. **Permission Errors**: Access denied, insufficient privileges
2. **Model Errors**: Model not found, incompatible versions
3. **Connection Errors**: Network issues, service unavailable
4. **Installation Errors**: Missing dependencies, configuration issues
5. **Execution Errors**: Command failures, invalid operations

## Data Flow

### Instruction Processing Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ Natural Language│
│   Processing    │ ──► Model Selection (Ollama/Gemini)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Command Planning│ ──► Command Generation
│   Phase 1       │ ──► Validation
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Command Execution│ ──► Terminal Operations
│   Phase 2       │ ──► Error Handling
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Result Verification│ ──► Success Validation
│   & Reporting   │ ──► Error Reporting
└─────────────────┘
    │
    ▼
User Output
```

## Configuration Architecture

### Configuration Hierarchy

```
┌─────────────────┐
│   config.yaml   │ (Primary configuration)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Environment     │ (Runtime overrides)
│ Variables       │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Command Line    │ (Session-specific)
│ Arguments       │
└─────────────────┘
```

### Key Configuration Sections

```yaml
api:
  preferred_provider: "ollama"
  local_endpoint: "http://localhost:11434"
  local_model: "gemini-3-flash-preview:latest"
  timeout: 120
  max_retries: 3

execution:
  safety_mode: true
  dry_run: false
  verify_commands: true

logging:
  level: "INFO"
  format: "structured"
  file: "logs/vexis.log"
```

## Security Architecture

### Security Layers

1. **Input Validation**: Sanitization and validation of user input
2. **Command Filtering**: Blacklist and whitelist for dangerous commands
3. **Permission Checking**: File system and execution permissions
4. **Sandboxing**: Isolated execution environment
5. **Audit Logging**: Complete audit trail of all operations

### Security Controls

```
┌─────────────────┐
│ Input Sanitizer │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Command Filter  │ ──► Blocked Commands
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Permission      │ ──► Access Denied
│ Checker         │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Execution       │
│ Sandbox         │
└─────────────────┘
```

## Performance Architecture

### Optimization Strategies

1. **Model Caching**: Cache model responses for similar queries
2. **Command Batching**: Group related commands for efficiency
3. **Parallel Execution**: Execute independent commands in parallel
4. **Resource Management**: Optimize memory and CPU usage
5. **Connection Pooling**: Reuse API connections

### Performance Metrics

- **Response Time**: < 2 seconds for simple commands
- **Throughput**: 10+ commands per minute
- **Memory Usage**: < 500MB baseline
- **CPU Usage**: < 25% during normal operation

## Extensibility Architecture

### Plugin System

```
┌─────────────────┐
│ Plugin Manager  │
└─────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Custom          │    │ AI Provider     │    │ Command         │
│ Commands        │    │ Plugins         │    │ Extensions      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Extension Points

1. **AI Providers**: Add new model providers
2. **Command Processors**: Custom command handling
3. **Error Handlers**: Specialized error handling
4. **Output Formatters**: Custom result formatting
5. **Security Modules**: Additional security layers

## Monitoring & Observability

### Monitoring Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Metrics          │    │ Logging         │    │ Health Checks   │
│ Collection      │    │ System          │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Performance     │    │ Audit Trail     │    │ System Status   │
│ Monitoring      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Metrics

- **Command Success Rate**: Percentage of successful executions
- **Response Latency**: Time from input to output
- **Error Rate**: Frequency of errors by type
- **Resource Utilization**: Memory, CPU, and network usage
- **Model Performance**: AI model response times and accuracy

## Future Architecture Considerations

### Scalability

- **Horizontal Scaling**: Multiple worker processes
- **Load Balancing**: Distribute workload across instances
- **Caching Layer**: Redis for response caching
- **Message Queue**: Async command processing

### High Availability

- **Health Monitoring**: Continuous health checks
- **Failover Mechanisms**: Automatic failover for critical components
- **Data Replication**: Backup and recovery systems
- **Disaster Recovery**: Complete system recovery procedures

### Advanced Features

- **Multi-Modal Support**: Image and audio processing
- **Workflow Engine**: Complex automation workflows
- **Collaboration Features**: Multi-user support
- **Advanced Analytics**: Usage analytics and insights
