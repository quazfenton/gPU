"""
Google Cloud Function generated from Kaggle notebook.
"""

import json
import logging
from flask import Flask, request, jsonify

# Optional model import if available
try:
    import deploy_model as _deploy_model

    _HAS_MODEL = True
except Exception:
    _deploy_model = None
    _HAS_MODEL = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Notebook code
#!/usr/bin/env python
# coding: utf-8

# # Simple Deployable API Example
#
# This notebook demonstrates how to create a simple API endpoint that can be deployed to:
# - Google Cloud Functions
# - AWS Lambda
# - Modal.com
#
# ## How to Deploy
#
# ```bash
# # Google Cloud Functions
# python3 runna.py deploy ./simple_api.ipynb --gcp-project YOUR_PROJECT
#
# # AWS Lambda
# python3 runna.py deploy-aws ./simple_api.ipynb --role-arn YOUR_ROLE_ARN
#
# # Modal.com
# python3 runna.py deploy-modal ./simple_api.ipynb
# ```

# ## Step 1: Import Dependencies
#
# Keep imports simple and standard. The deployment tool will automatically detect common packages.


import json
import math
from datetime import datetime

# ## Step 2: Define Your API Logic
#
# The `process_request` function is the entry point for your API.
# It receives a dictionary and should return a dictionary.


def process_request(data):
    """
    Main API handler function.

    Args:
        data (dict): Input data from POST request

    Returns:
        dict: Response data
    """
    try:
        # Get operation from request
        operation = data.get("operation", "info")

        if operation == "info":
            return {
                "status": "success",
                "message": "Simple API Example",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "available_operations": ["info", "calculate", "reverse"],
            }

        elif operation == "calculate":
            # Perform a calculation
            numbers = data.get("numbers", [])
            if not numbers:
                return {"error": "Please provide numbers array"}

            result = {
                "sum": sum(numbers),
                "mean": sum(numbers) / len(numbers),
                "max": max(numbers),
                "min": min(numbers),
                "count": len(numbers),
            }
            return {"status": "success", "result": result}

        elif operation == "reverse":
            # Reverse a string
            text = data.get("text", "")
            if not text:
                return {"error": "Please provide text to reverse"}

            return {"status": "success", "original": text, "reversed": text[::-1]}

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": ["info", "calculate", "reverse"],
            }

    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}


# ## Step 3: Test Locally
#
# Always test your function before deploying!


# Test 1: Info
print("Test 1: Info")
result = process_request({"operation": "info"})
print(json.dumps(result, indent=2))
print()


# Test 2: Calculate
print("Test 2: Calculate")
result = process_request({"operation": "calculate", "numbers": [10, 20, 30, 40, 50]})
print(json.dumps(result, indent=2))
print()


# Test 3: Reverse
print("Test 3: Reverse")
result = process_request({"operation": "reverse", "text": "Hello World!"})
print(json.dumps(result, indent=2))
print()


# Test 4: Error handling
print("Test 4: Error Handling")
result = process_request(
    {
        "operation": "calculate",
        "numbers": [],  # Empty array should trigger error message
    }
)
print(json.dumps(result, indent=2))

# ## Next Steps
#
# After deploying, you can call your API:
#
# ```bash
# # Using curl
# curl -X POST https://your-endpoint.com \
#   -H "Content-Type: application/json" \
#   -d '{"operation": "calculate", "numbers": [1,2,3,4,5]}'
#
# # Using the CLI tool
# python3 runna.py call my-endpoint --json '{"operation": "info"}'
# ```


def predict_handler(request):
    """HTTP Cloud Function entry point."""
    try:
        # Handle CORS
        if request.method == "OPTIONS":
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "3600",
            }
            return ("", 204, headers)

        # Set CORS headers for main request
        headers = {"Access-Control-Allow-Origin": "*"}

        if request.method == "POST":
            request_json = request.get_json(silent=True)
            if request_json is None:
                return jsonify({"error": "No JSON data provided"}), 400, headers
            # Use notebook-defined processing if present; otherwise call deploy_model.predict if available; else echo
            try:
                if "process_request" in globals() and callable(
                    globals()["process_request"]
                ):
                    result = globals()["process_request"](request_json)
                elif _HAS_MODEL and hasattr(_deploy_model, "predict"):
                    # Minimal example: expects features in JSON under "features"
                    features = request_json.get("features")
                    if features is None:
                        return (
                            jsonify({"error": 'Missing "features" in request body'}),
                            400,
                            headers,
                        )
                    # This is a placeholder; real model would require proper loading
                    result = {"prediction": _deploy_model.predict(features)}
                else:
                    result = {"echo": request_json}
            except Exception as inner_e:
                logger.exception("Processing error")
                return jsonify({"error": str(inner_e)}), 500, headers
            return jsonify(result), 200, headers
        elif request.method == "GET":
            return jsonify({"status": "Kaggle notebook API is running"}), 200, headers
        else:
            return jsonify({"error": "Method not allowed"}), 405, headers
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500, headers


# For local testing
if __name__ == "__main__":
    app = Flask(__name__)
    app.add_url_rule("/", "predict_handler", predict_handler, methods=["GET", "POST"])
    app.run(debug=True, port=8080)
