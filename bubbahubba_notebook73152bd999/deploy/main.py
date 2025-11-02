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

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def predict_handler(request):
    """HTTP Cloud Function entry point."""
    try:
        # Handle CORS
        if request.method == 'OPTIONS':
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600'
            }
            return ('', 204, headers)

        # Set CORS headers for main request
        headers = {'Access-Control-Allow-Origin': '*'}

        if request.method == 'POST':
            request_json = request.get_json(silent=True)
            if request_json is None:
                return jsonify({'error': 'No JSON data provided'}), 400, headers
            # Use notebook-defined processing if present; otherwise call deploy_model.predict if available; else echo
            try:
                if 'process_request' in globals() and callable(globals()['process_request']):
                    result = globals()['process_request'](request_json)
                elif _HAS_MODEL and hasattr(_deploy_model, 'predict'):
                    # Minimal example: expects features in JSON under "features"
                    features = request_json.get('features')
                    if features is None:
                        return jsonify({'error': 'Missing "features" in request body'}), 400, headers
                    # This is a placeholder; real model would require proper loading
                    result = {'prediction': _deploy_model.predict(features)}
                else:
                    result = {'echo': request_json}
            except Exception as inner_e:
                logger.exception('Processing error')
                return jsonify({'error': str(inner_e)}), 500, headers
            return jsonify(result), 200, headers
        elif request.method == 'GET':
            return jsonify({'status': 'Kaggle notebook API is running'}), 200, headers
        else:
            return jsonify({'error': 'Method not allowed'}), 405, headers
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500, headers

# For local testing
if __name__ == '__main__':
    app = Flask(__name__)
    app.add_url_rule('/', 'predict_handler', predict_handler, methods=['GET', 'POST'])
    app.run(debug=True, port=8080)
