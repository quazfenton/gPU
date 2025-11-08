#!/bin/bash
# Test app library functionality

set -e

echo "=== App Library Test ==="

# List apps
echo -e "\n1. Listing apps..."
python3 runna.py app-list

# Show app
echo -e "\n2. Showing image-classifier..."
python3 runna.py app-show image-classifier | head -20

# Create test app
echo -e "\n3. Creating test app..."
cat > /tmp/test_app.py << 'EOF'
import modal

app = modal.App("test-app")
image = modal.Image.debian_slim().pip_install("numpy")

@app.function(image=image)
@modal.web_endpoint(method="POST")
def test(data: dict):
    import numpy as np
    return {"result": np.mean(data.get("values", [1,2,3]))}
EOF

# Add test app
echo -e "\n4. Adding test app..."
python3 runna.py app-add test-app \
  --file /tmp/test_app.py \
  --description "Test app" \
  --tags test

# List again
echo -e "\n5. Listing apps (should include test-app)..."
python3 runna.py app-list

# Update test app
echo -e "\n6. Updating test app..."
echo '# Updated' >> /tmp/test_app.py
python3 runna.py app-update test-app --file /tmp/test_app.py

# Delete test app
echo -e "\n7. Deleting test app..."
python3 runna.py app-delete test-app

# List final
echo -e "\n8. Final list (test-app should be gone)..."
python3 runna.py app-list

echo -e "\n=== All tests passed! ==="
