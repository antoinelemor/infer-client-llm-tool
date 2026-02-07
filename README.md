# infer-client-llm-tool

Python client SDK and Rich CLI for the [Transformer Inference API](https://github.com/antoinelemor/infer-api-llm-tool) and [Ollama](https://ollama.ai) — companion package to [LLM Tool](https://github.com/antoinelemor/LLM_Tool).

Classify texts at scale with models trained via LLM Tool, or generate text with local decoder LLMs via Ollama. Features a progress-tracked Rich interface matching the LLM Tool visual style.

## Ecosystem

| Repository | Description |
|------------|-------------|
| [LLM_Tool](https://github.com/antoinelemor/LLM_Tool) | Main pipeline: LLM annotation, BERT training, validation |
| [infer-api-llm-tool](https://github.com/antoinelemor/infer-api-llm-tool) | Multi-model FastAPI inference server |
| **infer-client-llm-tool** | Python SDK + CLI client (this repo) |
| [inferclientllmtool](https://github.com/antoinelemor/inferclientllmtool) | R client SDK |

## Features

### Classification & NER
- **Multi-model support** — target any model registered on the inference server
- **Multi-label classification** — supports binary, multi-class, multi-label, and one-vs-all training modes
- **Parallel GPU+CPU inference** — hybrid inference for large batches using server-side parallelization
- **Configurable thresholds** — customize multi-label prediction thresholds
- **MC Dropout confidence intervals** — estimate prediction uncertainty with per-class CI bounds
- **Zero-shot NER** — extract entities with custom labels using GLiNER (requires server with `pip install 'infer-api[ner]'`)
  - **Multilingual**: 12+ languages (EN, FR, DE, ES, IT, PT, NL, RU, ZH, JA, AR)
  - **Custom labels**: ANY entity type ("political party", "disease", "product", etc.)
  - **Credit**: [urchade/GLiNER](https://github.com/urchade/GLiNER) (third-party model)

### Generation & Integration
- **Ollama integration** — generate text and chat with local LLMs (llama3, mistral, phi3, etc.)
- **TranslateGemma** — translate text between 130+ languages via the server or locally
- **Rich CLI** — ASCII art banner, colored panels, progress bars with ETA and rows/s
- **CSV classification** — batch-process entire datasets with real-time progress tracking
- **DataFrame integration** — classify pandas DataFrames with automatic column enrichment
- **Probability columns** — per-class probabilities appended to output

## Quick start

Clone the repo and use `make` for a one-step install:

```bash
git clone https://github.com/antoinelemor/infer-client-llm-tool.git
cd infer-client-llm-tool
make install        # editable install
make run            # launch the interactive CLI
```

All available Make targets:

| Command | Description |
|---------|-------------|
| `make install` | Install the package in editable mode |
| `make install-dev` | Install with dev dependencies (pytest, pandas) |
| `make install-pandas` | Install with pandas support (CSV/DataFrame) |
| `make run` | Launch the interactive CLI |
| `make test` | Run the test suite |
| `make clean` | Remove build artifacts |

## Installation (pip)

```bash
pip install "infer-client[pandas] @ git+https://github.com/antoinelemor/infer-client-llm-tool.git"
```

Minimal (no CSV/DataFrame support):

```bash
pip install git+https://github.com/antoinelemor/infer-client-llm-tool.git
```

## Interactive CLI

Launch the CLI by running `infer` — no arguments needed:

```bash
infer
```

On first launch you are prompted for your API URL and key. Credentials are saved to `~/.infer_client/credentials.json` (permissions `0600`) and reused automatically on subsequent launches.

### Main menu

After connecting, an interactive menu lets you navigate all features:

```
1  Health Check — API status and loaded models
2  List Models — All available models
3  Model Info — Detailed metadata for a model
4  Classify Text — Run inference on text(s)
5  Classify CSV — Batch-classify an entire dataset
6  Ollama — Local LLM generation & chat
7  Credentials — View, change, or remove saved credentials
0  Exit
```

Every action is guided: you pick a model from a numbered list, enter texts line-by-line, browse to a CSV file with path validation, choose the text column, set batch size, and confirm before classification starts.

### Visual output

The CLI displays Rich-formatted output with:
- Centered ASCII banner matching the LLM Tool style
- Bordered panels for health status, file info, and completion summary
- Color-coded tables for models, inference results, and probability distributions
- Live progress bar with spinner, percentage, row count, elapsed time, and ETA
- Graceful Ctrl+C handling with exit confirmation

## Python SDK

```python
from infer_client import InferClient

client = InferClient(
    base_url="https://your-server.example.com",
    api_key="YOUR_API_KEY",
)
```

### Health check (no auth)

```python
client.health()
# {'status': 'ok', 'version': '2.0.0', 'models_count': 3, 'default_model': 'sentiment', ...}
```

### List available models (no auth)

```python
client.models()
# [{'model_id': 'sentiment', 'model_type': 'classification', 'training_mode': 'multi-class', ...}]
```

### Model metadata (no auth)

```python
info = client.model_info("sentiment")
# Full metadata: base_model, labels, metrics, hyperparameters, training_mode, multi_label, etc.
```

### Optimal inference configuration (no auth)

```python
config = client.model_config("sentiment", n_texts=1000)
# {'batch_size': 32, 'use_parallel': True, 'device_mode': 'both', 'training_mode': 'multi-class', ...}
```

### Server resources (no auth)

```python
client.resources()
# {'cpu': {'cores': 10, 'percent': 25.3}, 'memory': {'total_gb': 32, ...}, 'gpu': {...}}
```

### Inference

```python
# Default model
result = client.infer(text="The stock market is crashing")
print(result["results"][0]["label"])       # sentiment_long_negative
print(result["results"][0]["confidence"])   # 0.58

# Specific model
result = client.infer(text="Great earnings", model="sentiment")

# Batch
result = client.infer(texts=[
    "Le marché est en pleine crise",
    "The economy is stable",
    "Terrible losses this quarter",
])

# Parallel GPU+CPU inference (for large batches)
result = client.infer(texts=large_batch, model="sentiment", parallel=True, device_mode="both")

# Multi-label inference with custom threshold
result = client.infer(text="Economic policy affects markets", model="themes", threshold=0.3)
# result["results"][0]["labels"] = ["economy", "politics"]
# result["results"][0]["label_count"] = 2

# Shortcut — returns results list directly
results = client.classify("Great performance this year")
results = client.classify(["Good", "Bad", "Neutral"], model="sentiment")

# Inference with MC Dropout confidence intervals
result = client.infer(text="The economy is booming", model="sentiment", mc_samples=30, ci_level=0.95)
# result["results"][0]["confidence_interval"] = {"lower": 0.25, "upper": 0.97, "level": 0.95, "mc_samples": 30}
# result["results"][0]["probabilities_ci"]["sentiment_long_positive"] = {"mean": 0.61, "std": 0.18, "lower": 0.25, "upper": 0.97}

# Shortcut with CI
results = client.classify("Great news", model="sentiment", mc_samples=30)
```

### DataFrame classification (requires pandas)

```python
import pandas as pd

df = pd.DataFrame({"text": ["Good news", "Bad news", "Neutral statement"]})

# Single-label model
result_df = client.classify_df(df, "text", model="sentiment")
# Columns: text, label, confidence, prob_negative, prob_neutral, prob_positive

# Multi-label model
result_df = client.classify_df(df, "text", model="themes", threshold=0.3)
# Columns: text, labels (list), label_count, threshold, prob_economy, prob_politics, ...

# Parallel inference for large datasets
result_df = client.classify_df(df, "text", model="sentiment", parallel=True)

# With MC Dropout confidence intervals
result_df = client.classify_df(df, "text", model="sentiment", mc_samples=30)
# Adds ci_lower_*, ci_upper_*, ci_level, mc_samples columns
```

### Sentence Segmentation (WTPSPLIT)

```python
# Segment text into sentences (85+ languages)
results = client.segment_sentences(
    "First sentence. Second sentence! Third sentence?"
)

print(results[0]["sentences"])
# ['First sentence.', 'Second sentence!', 'Third sentence?']

# Batch processing
texts = [
    "Hello world. How are you?",
    "This is another text. With multiple sentences."
]
results = client.segment_sentences(texts)

# Preserve newlines mode
results = client.segment_sentences(
    "Paragraph one.\n\nParagraph two.",
    mode="newline"
)

# Access results
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentences ({result['sentence_count']}):")
    for sentence in result['sentences']:
        print(f"  - {sentence}")
```

**Features**:
- Fast multilingual sentence boundary detection (85+ languages)
- Modes: `sentence` (default) or `newline` (preserves paragraph structure)
- Small model (3 layers) with high accuracy
- Credit: [WTPSPLIT](https://github.com/segment-any-text/wtpsplit) (segment-any-text/sat-3l-sm)

### Named Entity Recognition (requires server with NER support)

```python
# Extract entities with standard labels
results = client.extract_entities(
    "Apple Inc. was founded by Steve Jobs in Cupertino",
    labels=["person", "organization", "location"]
)

print(results[0]["entities"])
# [
#   {"text": "Apple Inc.", "label": "organization", "start": 0, "end": 10, "score": 0.91},
#   {"text": "Steve Jobs", "label": "person", "start": 26, "end": 36, "score": 0.99},
#   {"text": "Cupertino", "label": "location", "start": 40, "end": 49, "score": 0.99}
# ]

# Extract custom entity types (zero-shot)
results = client.extract_entities(
    "The Democratic Party won against the Republican Party",
    labels=["political party", "election event"]
)

# Multilingual extraction (12+ languages)
results = client.extract_entities(
    "Emmanuel Macron est président de la France",
    labels=["person", "country", "job title"],
    threshold=0.4  # Adjust confidence threshold
)

# Batch processing
texts = [
    "OpenAI released GPT-4",
    "Microsoft acquired GitHub"
]
results = client.extract_entities(
    texts,
    labels=["company", "product", "event"]
)

# Access entities
for result in results:
    print(f"Text: {result['text']}")
    print(f"Found {result['entity_count']} entities:")
    for entity in result['entities']:
        print(f"  - {entity['text']} ({entity['label']}, {entity['score']:.2f})")
```

**Output format**: JSON list of results, each containing `text`, `entities` (list), `entity_count`, `labels_used`, `threshold`.

**Credit**: Uses [GLiNER](https://github.com/urchade/GLiNER) or [GLiNER2](https://huggingface.co/fastino/gliner2-large-v1) (third-party models, not LLM Tool trained)

**Note**: GLiNER2 supports 2048 token context (4x larger than GLiNER v1) for longer documents. Use `model="gliner2"` to access.

### CSV classification (requires pandas)

```python
output_path = client.classify_csv("data.csv", text_column="text")
# Writes data_classified.csv with label, confidence, and probability columns

output_path = client.classify_csv("data.csv", "text", output_path="results.csv", model="sentiment")

# With parallel inference
output_path = client.classify_csv("data.csv", "text", model="sentiment", parallel=True)
```

## Ollama SDK

```python
from infer_client import OllamaClient

# Connect to Ollama (default: localhost:11434)
ollama = OllamaClient()

# List available models
ollama.models()

# Generate text
response = ollama.generate("llama3", "What is machine learning?")
print(response)

# With system prompt
response = ollama.generate("llama3",
    prompt="Extract key entities from: The Federal Reserve raised interest rates",
    system="You are an NLP expert. Return only a comma-separated list."
)

# Chat (multi-turn)
result = ollama.chat("llama3", [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Python in one sentence."}
])
print(result["response"])

# Continue the conversation
messages = result["messages"]
messages.append({"role": "user", "content": "And R?"})
result = ollama.chat("llama3", messages)
```

## Translation (TranslateGemma)

### Server-side translation (via API)

```python
# Translate English to French
result = client.translate("The economy is growing steadily", source_lang="en", target_lang="fr")
print(result["translation"])  # "L'économie croît de manière constante"

# Translate with regional variant
result = client.translate("Hello world", source_lang="en", target_lang="zh-Hans")

# List supported languages
languages = client.translate_languages()
print(f"{len(languages)} languages supported")
```

### Local translation (direct Ollama)

```python
from infer_client import OllamaClient

ollama = OllamaClient()

# Translate locally
result = ollama.translate("Bonjour le monde", source_lang="fr", target_lang="en")
print(result["translation"])  # "Hello world"

# List supported languages
languages = OllamaClient.translate_languages()
```

## Response format

### Single-label inference response

```json
{
  "results": [
    {
      "text": "The economy is booming",
      "label": "sentiment_long_positive",
      "confidence": 0.7123,
      "probabilities": {
        "sentiment_long_negative": 0.1045,
        "sentiment_long_neutral": 0.1832,
        "sentiment_long_positive": 0.7123
      }
    }
  ],
  "count": 1,
  "model_type": "classification",
  "model_id": "sentiment",
  "training_mode": "multi-class",
  "multi_label": false,
  "num_labels": 3,
  "labels": ["sentiment_long_negative", "sentiment_long_neutral", "sentiment_long_positive"]
}
```

### With MC Dropout confidence intervals (`mc_samples > 0`)

```json
{
  "results": [
    {
      "text": "The economy is booming",
      "label": "sentiment_long_positive",
      "confidence": 0.6076,
      "probabilities": { ... },
      "confidence_interval": {
        "lower": 0.2498,
        "upper": 0.9654,
        "level": 0.95,
        "mc_samples": 30
      },
      "probabilities_ci": {
        "sentiment_long_negative": {"mean": 0.2262, "std": 0.1164, "lower": 0.0, "upper": 0.4543},
        "sentiment_long_neutral": {"mean": 0.1662, "std": 0.0813, "lower": 0.0069, "upper": 0.3254},
        "sentiment_long_positive": {"mean": 0.6076, "std": 0.1825, "lower": 0.2498, "upper": 0.9654}
      }
    }
  ]
}
```

### Multi-label inference response

```json
{
  "results": [
    {
      "text": "Economic policy affects markets",
      "labels": ["economy", "politics"],
      "label_count": 2,
      "threshold": 0.3,
      "probabilities": {
        "economy": 0.89,
        "politics": 0.72,
        "markets": 0.25,
        "sports": 0.03
      }
    }
  ],
  "count": 1,
  "model_type": "classification",
  "model_id": "themes",
  "training_mode": "multi-label",
  "multi_label": true,
  "multi_label_threshold": 0.3,
  "num_labels": 4,
  "labels": ["economy", "politics", "markets", "sports"]
}
```

### Model info response

`client.model_info(model_id)` returns comprehensive metadata including training metrics.

| Field | Description |
|-------|-------------|
| `model_id` | Model identifier |
| `base_model` | HuggingFace base model name |
| `training_mode` | Training mode: `binary`, `multi-class`, `multi-label`, `one-vs-all` |
| `multi_label` | Whether model uses multi-label classification |
| `multi_label_threshold` | Threshold for multi-label predictions (default: 0.5) |
| `labels` | List of class labels |
| `languages` | Supported languages |
| `epoch` | Number of training epochs |
| `timestamp` | Training completion timestamp |
| `training_time_seconds` | Total training time |
| `metrics` | Overall metrics: `macro_f1`, `accuracy`, `train_loss`, `val_loss` |
| `per_class_metrics` | Per-class `precision`, `recall`, `f1`, `support` |
| `per_language_metrics` | Per-language breakdown with metrics per class |
| `hyperparameters` | Training hyperparameters (learning_rate, batch_size, etc.) |
| `raw_config` | Full HuggingFace config.json contents |
| `raw_training_metadata` | Full LLM Tool training_metadata.json contents |

## Running tests

```bash
make install-dev
INFER_KEY="YOUR_API_KEY" make test
```

## License

MIT
