# Chat & Send Guide

Send text to deployed endpoints and receive responses.

## Quick Start

```bash
# Deploy LLM chat app
python3 runna.py app-deploy llm-chat

# Send single message
python3 runna.py send llm-chat "Hello, how are you?"

# Interactive chat
python3 runna.py chat llm-chat
```

## Commands

### send - Send Single Message

```bash
python3 runna.py send <endpoint> <text>
```

**Examples:**
```bash
# Send to saved endpoint
python3 runna.py send llm-chat "Tell me a joke"

# Send to URL directly
python3 runna.py send https://user--app.modal.run "Hello"

# Custom field name
python3 runna.py send llm-chat "Hello" --field prompt
```

**Output:**
```json
{
  "text": "Tell me a joke",
  "response": "Why did the chicken cross the road? To get to the other side!"
}
```

### chat - Interactive Chat

```bash
python3 runna.py chat <endpoint>
```

**Example:**
```bash
python3 runna.py chat llm-chat
```

**Session:**
```
Chatting with llm-chat (Ctrl+C to exit)
--------------------------------------------------

You: Hello!
Bot: Hello! How can I help you today?

You: Tell me about AI
Bot: AI stands for Artificial Intelligence...

You: ^C
Goodbye!
```

## LLM Chat App

Deploy the built-in LLM chat app:

```bash
# Deploy
python3 runna.py app-deploy llm-chat

# Chat
python3 runna.py chat llm-chat
```

**Features:**
- GPT-2 model
- GPU acceleration (A10G)
- Configurable max_length and temperature
- Streaming responses

## Custom Field Names

Different endpoints use different field names:

```bash
# Default: "text"
python3 runna.py send my-app "Hello"
# Sends: {"text": "Hello"}

# Custom field
python3 runna.py send my-app "Hello" --field prompt
# Sends: {"prompt": "Hello"}

# For chat
python3 runna.py chat my-app --field message
```

## Response Formats

The tool automatically extracts responses from common formats:

```json
{"response": "text"}      → "text"
{"text": "text"}          → "text"
{"output": "text"}        → "text"
{"result": {...}}         → JSON dump
```

## Examples

### Text Generation

```bash
# Deploy
python3 runna.py app-deploy text-generator

# Generate
python3 runna.py send text-generator "Once upon a time"
```

### Sentiment Analysis

```bash
# Create sentiment app
cat > sentiment.py << 'EOF'
import modal

app = modal.App("sentiment")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def analyze(data: dict):
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    result = classifier(data["text"])[0]
    return {"text": data["text"], "response": f"{result['label']} ({result['score']:.2f})"}
EOF

# Add and deploy
python3 runna.py app-add sentiment --file sentiment.py --gpu T4
python3 runna.py app-deploy sentiment

# Test
python3 runna.py send sentiment "I love this!"
```

### Question Answering

```bash
# Create QA app
cat > qa.py << 'EOF'
import modal

app = modal.App("qa")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def answer(data: dict):
    from transformers import pipeline
    qa = pipeline("question-answering")
    result = qa(question=data["text"], context=data.get("context", ""))
    return {"response": result["answer"]}
EOF

# Deploy and use
python3 runna.py app-add qa --file qa.py --gpu T4
python3 runna.py app-deploy qa
python3 runna.py send qa "What is AI?"
```

## Programmatic Usage

Use in Python scripts:

```python
from runna import send_text

# Send message
result = send_text("llm-chat", "Hello!")
print(result["response"])

# With custom field
result = send_text("my-app", "Hello", field="prompt")
```

## Batch Processing

Send multiple messages:

```bash
# Create batch script
cat > batch_send.sh << 'EOF'
#!/bin/bash
while IFS= read -r line; do
    python3 runna.py send llm-chat "$line"
done < messages.txt
EOF

chmod +x batch_send.sh
./batch_send.sh
```

## Tips

1. **Use saved endpoints** - Easier than typing URLs
2. **Check response format** - Different apps return different formats
3. **Set timeouts** - LLMs can take time to respond
4. **Use chat for testing** - Interactive mode is great for experimentation
5. **Custom fields** - Match your app's expected input format

## Troubleshooting

### Endpoint not found
```bash
# List endpoints
python3 runna.py endpoints

# Use full URL
python3 runna.py send https://full-url.modal.run "text"
```

### Timeout errors
```bash
# Increase timeout in app deployment
python3 runna.py app-deploy llm-chat --timeout 900
```

### Wrong field name
```bash
# Check app code for expected field
python3 runna.py app-show llm-chat

# Use correct field
python3 runna.py send llm-chat "text" --field prompt
```

## Integration

### With Endpoints
```bash
# Deploy and save
python3 runna.py app-deploy llm-chat

# Send immediately
python3 runna.py send llm-chat "Hello"
```

### With Scripts
```bash
# In bash
RESPONSE=$(python3 runna.py send llm-chat "Hello" | jq -r '.response')
echo "Bot said: $RESPONSE"
```

### With Pipelines
```bash
# Chain commands
echo "Tell me a joke" | xargs python3 runna.py send llm-chat
```

## See Also

- [APP_LIBRARY.md](APP_LIBRARY.md) - App library documentation
- [MODAL_GUIDE.md](MODAL_GUIDE.md) - Modal deployment guide
- [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) - Complete workflows
