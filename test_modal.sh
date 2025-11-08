#!/bin/bash
# Test script for Modal deployment

set -e

echo "=== Modal Deployment Test ==="

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check authentication
echo "Checking Modal authentication..."
if ! modal token set --help &> /dev/null; then
    echo "⚠️  Modal authentication required. Run: modal token new"
    exit 1
fi

echo "✅ Modal CLI ready"

# Create test notebook
TEST_DIR="test_modal_deploy"
mkdir -p "$TEST_DIR"

cat > "$TEST_DIR/test_notebook.py" << 'EOF'
"""Test notebook for Modal deployment."""
import numpy as np

def process_request(data):
    """Simple test function."""
    features = data.get('features', [1, 2, 3])
    result = np.mean(features)
    return {
        'result': float(result),
        'count': len(features),
        'status': 'success'
    }

if __name__ == '__main__':
    test_data = {'features': [1, 2, 3, 4, 5]}
    print(process_request(test_data))
EOF

echo "✅ Created test notebook"

# Test basic deployment
echo ""
echo "Testing basic deployment..."
python3 runna.py deploy-modal "$TEST_DIR" --save-name test-modal

if [ $? -eq 0 ]; then
    echo "✅ Basic deployment successful"
else
    echo "❌ Basic deployment failed"
    exit 1
fi

# Test with GPU option (won't actually use GPU without real model)
echo ""
echo "Testing GPU deployment..."
python3 runna.py deploy-modal "$TEST_DIR" --gpu T4 --save-name test-modal-gpu

if [ $? -eq 0 ]; then
    echo "✅ GPU deployment successful"
else
    echo "❌ GPU deployment failed"
    exit 1
fi

# Test with timeout
echo ""
echo "Testing custom timeout..."
python3 runna.py deploy-modal "$TEST_DIR" --timeout 600 --save-name test-modal-timeout

if [ $? -eq 0 ]; then
    echo "✅ Timeout configuration successful"
else
    echo "❌ Timeout configuration failed"
    exit 1
fi

echo ""
echo "=== All tests passed! ==="
echo ""
echo "Cleanup: rm -rf $TEST_DIR"
echo "View logs: modal app logs kaggle-notebook"
echo "List endpoints: python3 runna.py endpoints"
