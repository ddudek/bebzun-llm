# Bebzun-llm: Generating codebase knowledge for LLMs

Bebzun is a playful Polish word for a big, round belly—usually describing something that’s grown a bit too much, or is adorably oversized. Kind of like our project’s codebase sometimes.

The project explores a different approach to context engineering for large codebases. Instead of feeding raw source code to LLMs, this project focuses on extensive pre-processing of codebase files to create summarizations and build a knowledge base of the entire project that's easy to lookup and query.

It currently supports my daily work - Android, so these languages:
- Java
- Kotlin

In progress 🚧:
- Objective-C
- Swift

Tested primarily on Apple Mac with the MLX engine, but also supports Ollama.

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
[Retrieval] → Vector similarity search + BM25 + Reranker
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

This knowledge base can be used for various tasks, here explored:
- Simple chat about the codebase
- Generate initial context for more advanced 3rd party coding agent.
- Generate a comprehensive project summary/analysis for structure and all features, by listing all summaries (only for mid/smaller projects, ~1200 classes is roughly 128k tokens).
- Scripting with LLM processing by providing good initial project context - e.g. processing tickets for the development team.

## Features
- **Context-Engineered Summaries**: LLM receives dependency information and usage patterns for each class
- **Small-LLM Optimization**: Pre-processed summaries for efficient context usage. Supports Ollama, MLX-LM, with e.g. Qwen3 14b, Gemma3 etc. Also added Anthropic API for external provider.
- **Interactive Knowledge Query**: RAG-based Q&A system with vector and semantic search
- **Dependency-Order Processing**: Files processed in optimal order for maximum context
- **Incremental Processing**: Skip already processed files for efficiency
- **Vector-Based Search**: Find relevant code using natural language queries

## 1. Installation
```bash
pip install -r requirements.txt
```
Make sure you have local Ollama, or MLX-LM setup.

## 2. Configuration
Create configuration files at `<your-project>/.ai-agent/`:
`config.json` - use example/config_mlx.json or example/config_ollama.json
`project_context.txt` - general description of your project for initial processing

In `config.json` specify which directories contain your source code:
```json
{
  "source_dirs": [
    "app/src/main/java",
    "library/src/main/kotlin"
  ]
}
```

## 3. Build knowledge
Create a knowledge database for your project:
```bash
python build_knowledge.py -i /path/to/your/project
```

### Step 1: Static Analysis
Analyzes Java and Kotlin source files to extract class structures, methods, and dependencies. Creates `<your-project>/.ai-agent/db_preprocess.json`.

You can also run this step manually by adding `-m Pre` parameter:
```bash
python build_knowledge.py -i /path/to/your/project -m Pre
```

### Step 2: Knowledge Building
Uses LLM to generate summaries of classes and methods with contextual information. Creates `ai-agent/db_final.json` with LLM summaries and embeddings database. This step can take significant amount of time. Macbook M1 Pro with 32GB RAM and using Qwen3-14b, will take 2 days for processing a project with ~1300 files.
Database is created incrementally, using already processed files as a context for the next ones. After each processed file, output file is updated to avoid loosing progress.

You can also run this step manually by adding `-m Final` parameter:
```bash
python build_knowledge.py -i /path/to/your/project -m Final [-f "MainActivity"]
```

### Step 3: Create embeddings for search
Creates and stores embeddings for each piece of knowledge.

You can also run this step manually by adding `-m Embedd` parameter:
```bash
python build_knowledge.py -i /path/to/your/project -m Embedd
```

### Step 2: Interaction
The project is ready for interaction:

Chat:
```bash
python chat.py -i /path/to/your/project
```

CLI:

```bash
cli.py [-h] -i INPUT_DIR [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--log-file LOG_FILE] [--llm-log-file LLM_LOG_FILE] {similarity_search,bm25_search,summarize_project}

python cli.py -i /path/to/your/project similarity_search "Update status interval"
python cli.py -i /path/to/your/project bm25_search "Update status interval"
python cli.py -i /path/to/your/project summarize_project
```

## Output Files

The system uses simple, json-based storage, and generates output files in the `.ai-agent/` directory:

- `config.json`: Configuration file (user-provided)
- `db_preprocess.json`: Static analysis results
- `db_final.json`: LLM summaries and processed data
- `db_embeddings.json`: Vector embeddings file

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`
- LLM backend (Ollama server, MLX-LM setup, or Anthropic API (in progress 🚧) )

## Testing

```bash
# Run Java parser tests
python -m unittest static_analysis.tests.test_java_parser -v

# Run Kotlin parser tests  
python -m unittest static_analysis.tests.test_kotlin_parser -v