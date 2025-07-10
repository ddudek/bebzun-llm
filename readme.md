# Codebase knowledge context for LLMs

The project explores a different approach to context engineering for large codebases. Instead of feeding raw source code to LLMs, this project focuses on heavy pre-processing of codebase files to create summarizations and build a knowledge base of the entire project that's easy to lookup and query.

It currently supports my daily work - Android, so these languages:
- Java
- Kotlin

Tested mainly on Apple mac, and MLX engine, but also supports Ollama.

**The main goal:** 
Maximize LLM context efficiency by providing pre-processed, contextual summaries rather than raw code. This allows smaller LLMs (e.g. 8-32b models with Ollama, MLX-LM, etc.) to effectively understand and work with huge, poorly-written codebases that would otherwise saturate context quickly and cause issues with finding relevant code.

## Core Flow: Context Engineering Through Pre-Processing

```
[Raw Source Code] 
    ↓
[Static Analysis] → Extract class structures, methods, dependencies (Tree sitter) → db_preprocess.json
    ↓
[LLM Processing] → LLM processes each class with:
    │              • Usage patterns from other classes  
    │              • Already processed dependency class summaries
    │              • Method and variable contexts
    │              • Create summaries of class content, usages, features, methods, properties
    │            → db_final.json
    ↓
[Knowledge Base] → LLM summaries + vector embeddings → db_embeddings.json
    ↓  
[Retrieval] → Vector similarity search + BM25
    ↓  
[Interactive Query] → Simple RAG-based chat with pre-processed knowledge
```

## Static analysis
Tree-sitter extracts class structure information, including dependencies. It identifies not only full class name references (e.g., in constructors or static references) but also variable/property types and their usages, saving them as dependencies. This information is then used to provide context about both dependencies and their usage to the LLM during processing.

Example output:
```json
    "com.example.MainActivityUiState": {
      "simple_classname": "MainActivityUiState",
      "full_classname": "com.google.samples.apps.nowinandroid.MainActivityUiState",
      "dependencies": [
        {
          "simple_classname": "UserData",
          "full_classname": "com.example.model.data.UserData",
          "usage_lines": [
            50,
            51,
            53,
            59
          ]
        },
        {
          "simple_classname": "Loading",
          "full_classname": "com.example.MainActivityUiState.Loading",
          "usage_lines": [
            69
          ]
        }
      ],
      "public_methods": [
        {
          "name": "shouldUseDarkTheme",
          "definition_start": 58,
          "definition_end": 63
        },
        {
          "name": "shouldKeepSplashScreen",
          "definition_start": 69,
          "definition_end": 69
        },
        {
          "name": "shouldUseDarkTheme",
          "definition_start": 84,
          "definition_end": 84
        }
      ],
      "source_file": "app/src/main/kotlin/com/example/MainActivityViewModel.kt"
    },
```
## LLM Processing
After static analysis, the LLM processes each class to generate summaries. It uses the dependency and usage information to understand the class's role in the codebase. The output includes a summary of the class, its category (e.g., Logic, UI), potential questions it can answer, key features, and summaries of its methods and variables. This structured knowledge is what powers the retrieval and query system.

Example:
```json
"classes": [
    {
      "simple_classname": "MainActivityViewModel",
      "full_classname": "com.example.MainActivityViewModel",
      "summary": "MainActivityViewModel is a ViewModel that manages the UI state for the main activity. It observes user data from UserDataRepository and transforms it into a UI state that can be used by the UI layer. The ViewModel ensures that the UI is updated with the latest user data, such as theme preferences and followed topics, and provides a consistent way to handle state changes across the app.",
      "category": "Logic",
      "questions": [
        "How does the app handle theme preference changes when the user selects a new theme?",
        "What happens when the user updates their followed topics through the UI?",
        "How does the app ensure that the UI reflects the latest user data changes in real-time?"
      ],
      "features": [
        "UI state management",
        "User data observation",
        "Theme preference handling"
      ],
      "methods": [
        {
          "method_name": "constructor",
          "method_summary": "The constructor initializes the ViewModel with a UserDataRepository instance. It sets up the uiState flow by mapping the user data to a UI state, ensuring that the UI is updated with the latest user data changes. The flow is maintained within the ViewModel's scope, providing real-time updates."
        }
      ],
      "variables": [
        {
          "variable_name": "uiState",
          "variable_summary": "uiState is a StateFlow<MainActivityUiState> that provides the current UI state to the UI layer. It is derived from the user data observed from UserDataRepository. The flow is updated whenever user data changes, ensuring that the UI reflects the latest state. The initial value is Loading, and the flow is maintained within the ViewModel's scope."
        }
      ]
    },
]
```

Then this learned knowledge can be used for various tasks, here explored:
- Simple chat about the codebase
- Generate initial context for more advanced 3rd party coding agent.
- Generate a comprehensive project summary/analysis for structure and all features, by listing all summaries (only for mid/smaller projects, ~1200 classes is roughly 128k tokens).
- Scripting with LLM processing by providing good initial project context - e.g. processing tickets for the development team.

## Features
- **Context-Engineered Summaries**: LLM receives dependency information and usage patterns for each class
- **Small-LLM Optimization**: Pre-processed summaries for efficient context usage. Supports Ollama, MLX-LM, with e.g. Qwen3 14b, Gemma3 etc. Also added Anthropic API for external provider.
- **Interactive Knowledge Query**: RAG-based Q&A system with semantic search
- **Dependency-Order Processing**: Files processed in optimal order for maximum context
- **Incremental Processing**: Skip already processed files for efficiency
- **Vector-Based Search**: Find relevant code using natural language queries

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a configuration file at `<your-project>/.ai-agent/config.json`:

```json
{
  "source_dirs": ["app/src/main/java"],
  "llm": {
    "mode": "ollama",
    "ollama": {
      "url": "http://localhost:11434",
      "model": "gemma3:12b",
      "temperature": 0.7
    }
  }
}
```

### 3. Run Analysis
```bash
# Step 1: Static Analysis
python static_analysis.py -i /path/to/your/project

# Step 2: Knowledge Building
python build_knowledge.py -i /path/to/your/project -m Final
python build_knowledge.py -i /path/to/your/project -m Embedd

# Step 3: Interactive Chat
python chat.py -i /path/to/your/project
```

## Usage

### Static Analysis
Analyzes Java and Kotlin source files to extract class structures, methods, and dependencies.

```bash
python static_analysis.py -i <input_directory>
```

**Options:**
- `-i, --input-dir`: Directory containing Java/Kotlin source files (required)

**Output:** `.ai-agent/db_preprocess.json` containing class structures and dependencies

### Knowledge Building
Uses LLM to generate intelligent summaries of classes and methods with contextual information.

```bash
python build_knowledge.py -i <input_directory> -m <mode> [-filter <pattern>]
```

**Options:**
- `-i, --input-dir`: Project directory (required)
- `-m, --mode`: Processing mode - `Pre`, `Final`, or `Embedd`
- `-filter`: Filter files by name (prefix with `!` to exclude)

**Processing Modes:**
- `Final`: LLM summarization with dependency context
- `Embedd`: Generate embeddings for semantic search

**Output:** `.ai-agent/db_final.json` with LLM summaries and embeddings database

### Interactive Chat
Provides an interactive Q&A interface with access to analyzed codebase.

```bash
python -m interact.chat <project_directory>
```

**Features:**
- Natural language queries about code
- Semantic search across codebase
- Context-aware responses using RAG
- File content retrieval and analysis

## Configuration

### LLM Backend Options

**Ollama (Local):**
```json
{
  "llm": {
    "mode": "ollama",
    "ollama": {
      "url": "http://localhost:11434",
      "model": "gemma3:12b",
      "temperature": 0.7
    }
  }
}
```

**MLX (Apple Silicon):**
```json
{
  "llm": {
    "mode": "mlx",
    "mlx": {
      "model": "mlx-community/gemma-3-12b-it-qat-4bit",
      "temperature": 0.7
    }
  }
}
```

**Anthropic Claude:**
```json
{
  "llm": {
    "mode": "anthropic",
    "anthropic": {
      "key": "YOUR_ANTHROPIC_API_KEY",
      "model": "claude-3-5-sonnet-latest"
    }
  }
}
```

### Source Directories
Specify which directories contain your source code:

```json
{
  "source_dirs": [
    "app/src/main/java",
    "library/src/main/kotlin"
  ]
}
```

## Output Files

The system uses simple, json-based storage, and generates several output files in the `.ai-agent/` directory:

- `config.json`: Configuration file (user-provided)
- `db_preprocess.json`: Static analysis results
- `db_final.json`: LLM summaries and processed data
- `db_embeddings.json`: Vector embeddings file

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- LLM backend (Ollama server, MLX-LM setup, or Anthropic API)

## Testing

```bash
# Run Java parser tests
python -m unittest static_analysis.tests.test_java_parser -v

# Run Kotlin parser tests  
python -m unittest static_analysis.tests.test_kotlin_parser -v