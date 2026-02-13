# Send/Chat Implementation Summary

Added text input/output functionality for deployed serverless endpoints.

## What Was Added

### Core Functions (runna.py)

**send_text()** - Send text to endpoint and get response
```python
def send_text(endpoint: str, text: str, field: str = "text") -> dict
```

### CLI Commands

**send** - Send single message
```bash
python3 runna.py send <endpoint> <text> [--field <name>]
```

**chat** - Interactive chat session
```bash
python3 runna.py chat <endpoint> [--field <name>]
```

### New App Template

**llm-chat** - Interactive LLM chat endpoint
- GPT-2 model
- GPU: A10G
- Configurable max_length and temperature
- Returns formatted responses

### Documentation

- **CHAT_GUIDE.md** - Complete chat/send documentation
- **test_send.sh** - Test script
- Updated **README.md** with send/chat section
- Updated **apps/QUICKREF.md** with new commands

## Usage

### Send Single Message

```bash
# Deploy LLM
python3 runna.py app-deploy llm-chat

# Send message
python3 runna.py send llm-chat "Hello, how are you?"
```

Output:
```json
{
  "text": "Hello, how are you?",
  "response": "I'm doing well, thank you for asking!"
}
```

### Interactive Chat

```bash
python3 runna.py chat llm-chat
```

Session:
```
Chatting with llm-chat (Ctrl+C to exit)
--------------------------------------------------

You: Tell me a joke
Bot: Why did the chicken cross the road? To get to the other side!

You: That's funny
Bot: I'm glad you enjoyed it!

You: ^C
Goodbye!
```

### Custom Field Names

```bash
# Default field: "text"
python3 runna.py send my-app "Hello"
# Sends: {"text": "Hello"}

# Custom field
python3 runna.py send my-app "Hello" --field prompt
# Sends: {"prompt": "Hello"}
```

## Features

### Automatic Response Extraction

The tool automatically extracts responses from common formats:
- `{"response": "text"}` → "text"
- `{"text": "text"}` → "text"
- `{"output": "text"}` → "text"
- Other formats → JSON dump

### Endpoint Resolution

Works with both saved endpoints and URLs:
```bash
# Saved endpoint
python3 runna.py send llm-chat "Hello"

# Direct URL
python3 runna.py send https://user--app.modal.run "Hello"
```

### Error Handling

- Validates endpoint exists
- Handles network errors
- Shows clear error messages
- Graceful Ctrl+C exit in chat mode

## LLM Chat App

New built-in app for interactive chat:

```python
import modal

app = modal.App("llm-chat")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

@app.function(image=image, gpu="A10G", timeout=600)
@modal.web_endpoint(method="POST")
def chat(data: dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    text = data.get("text", "")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0])
    
    return {"text": text, "response": response}
```

## Integration

### With App Library

```bash
# Deploy from library
python3 runna.py app-deploy llm-chat

# Send immediately
python3 runna.py send llm-chat "Hello"
```

### With Endpoints

```bash
# List endpoints
python3 runna.py endpoints

# Send to any endpoint
python3 runna.py send my-endpoint "text"
```

### Programmatic Usage

```python
from runna import send_text

result = send_text("llm-chat", "Hello!")
print(result["response"])
```

## Use Cases

### LLM Interaction
- Chat with language models
- Text generation
- Question answering
- Summarization

### API Testing
- Test deployed endpoints
- Validate responses
- Debug issues
- Performance testing

### Automation
- Batch processing
- Scripted interactions
- Pipeline integration
- Monitoring

## Examples

### Text Generation

```bash
python3 runna.py app-deploy text-generator
python3 runna.py send text-generator "Once upon a time"
```

### Sentiment Analysis

```bash
python3 runna.py send sentiment-analyzer "I love this product!"
```

### Question Answering

```bash
python3 runna.py send qa-bot "What is artificial intelligence?"
```

### Batch Processing

```bash
cat messages.txt | while read line; do
    python3 runna.py send llm-chat "$line"
done
```

## Command Reference

### send

```bash
python3 runna.py send <endpoint> <text> [options]

Options:
  --field <name>    JSON field name (default: text)
```

### chat

```bash
python3 runna.py chat <endpoint> [options]

Options:
  --field <name>    JSON field name (default: text)

Controls:
  Ctrl+C           Exit chat
```

## File Changes

### Modified
- **runna.py**
  - Added `send_text()` function
  - Added `cmd_send()` command
  - Added `cmd_chat()` command
  - Added argument parsers

### Created
- **apps/llm_chat.py** - LLM chat app template
- **CHAT_GUIDE.md** - Complete documentation
- **test_send.sh** - Test script
- **SEND_CHAT_SUMMARY.md** - This file

### Updated
- **apps/library.json** - Added llm-chat app
- **README.md** - Added send/chat section
- **apps/QUICKREF.md** - Added send/chat commands

## Testing

```bash
# Run test script
./test_send.sh

# Manual testing
python3 runna.py app-list
python3 runna.py app-show llm-chat
python3 runna.py app-deploy llm-chat
python3 runna.py send llm-chat "Hello"
python3 runna.py chat llm-chat
```

## Benefits

1. **Simple Interface** - Easy text input/output
2. **Interactive Mode** - Chat for testing
3. **Flexible** - Works with any endpoint
4. **Integrated** - Uses existing endpoint registry
5. **Minimal** - Clean, focused implementation

## Summary

Added complete text interaction system:
- ✅ 2 new CLI commands (send, chat)
- ✅ 1 core function (send_text)
- ✅ 1 new app template (llm-chat)
- ✅ Automatic response extraction
- ✅ Endpoint resolution
- ✅ Error handling
- ✅ Interactive chat mode
- ✅ Complete documentation

All minimal, production-ready code for serverless LLM interaction!
