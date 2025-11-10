# Send/Chat Quick Example

## 1. Deploy LLM Chat

```bash
python3 runna.py app-deploy llm-chat
```

## 2. Send Single Message

```bash
python3 runna.py send llm-chat "Tell me a joke"
```

Output:
```json
{
  "text": "Tell me a joke",
  "response": "Why did the chicken cross the road? To get to the other side!"
}
```

## 3. Interactive Chat

```bash
python3 runna.py chat llm-chat
```

```
Chatting with llm-chat (Ctrl+C to exit)
--------------------------------------------------

You: Hello!
Bot: Hello! How can I help you today?

You: What is AI?
Bot: AI stands for Artificial Intelligence...

You: Thanks!
Bot: You're welcome!

You: ^C
Goodbye!
```

## 4. Use with Any Endpoint

```bash
# List available endpoints
python3 runna.py endpoints

# Send to any endpoint
python3 runna.py send my-endpoint "Hello"

# Or use URL directly
python3 runna.py send https://user--app.modal.run "Hello"
```

## 5. Custom Field Names

```bash
# Some apps use different field names
python3 runna.py send my-app "Hello" --field prompt
python3 runna.py chat my-app --field message
```

## Complete Workflow

```bash
# 1. List apps
python3 runna.py app-list

# 2. Deploy
python3 runna.py app-deploy llm-chat

# 3. Test with send
python3 runna.py send llm-chat "Test message"

# 4. Interactive chat
python3 runna.py chat llm-chat

# 5. Check endpoints
python3 runna.py endpoints
```

That's it! Simple text interaction with deployed serverless LLMs.
