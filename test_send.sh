#!/bin/bash
# Test send/chat functionality

echo "=== Send/Chat Test ==="

# Create test endpoint
echo -e "\n1. Creating test endpoint..."
cat > /tmp/test_endpoint.py << 'EOF'
import modal

app = modal.App("test-endpoint")
image = modal.Image.debian_slim()

@app.function(image=image)
@modal.web_endpoint(method="POST")
def echo(data: dict):
    text = data.get("text", "")
    return {"text": text, "response": f"Echo: {text}"}
EOF

# Add to library
python3 runna.py app-add test-endpoint \
  --file /tmp/test_endpoint.py \
  --description "Test echo endpoint"

echo -e "\n2. Test send command (dry run - no actual deployment)..."
echo "Command: python3 runna.py send test-endpoint 'Hello World'"
echo "Expected: Would send {'text': 'Hello World'} to endpoint"

echo -e "\n3. Test chat command (dry run)..."
echo "Command: python3 runna.py chat test-endpoint"
echo "Expected: Interactive chat session"

echo -e "\n4. Cleanup..."
python3 runna.py app-delete test-endpoint

echo -e "\n=== Test Complete ==="
echo "To test with real deployment:"
echo "  1. python3 runna.py app-deploy llm-chat"
echo "  2. python3 runna.py send llm-chat 'Hello'"
echo "  3. python3 runna.py chat llm-chat"
