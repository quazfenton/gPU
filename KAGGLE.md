# Kaggle Notebook Automation & Deployment Tool

A comprehensive CLI tool for automating Kaggle notebook workflows and deploying them to multiple serverless platforms (Google Cloud Functions, AWS Lambda, Modal.com).

## 🚀 Features

### Core Functionality
- **Download/Pull** notebooks from Kaggle
- **Upload/Push** notebooks to Kaggle
- **Run** notebooks remotely on Kaggle infrastructure
- **Interactive Selection** for browsing popular kernels
- **Batch Processing** for multiple notebooks

### Deployment Platforms
- ✅ **Google Cloud Functions** - Deploy to GCP with automatic scaling
- ✅ **AWS Lambda** - Deploy to AWS serverless functions
- ✅ **Modal.com** - Modern serverless Python platform
- ✅ **Local Server** - Test locally before deploying

### Advanced Features
- **Automatic Notebook Conversion** - Converts Jupyter notebooks to deployable Python code
- **Magic Command Filtering** - Removes IPython/Jupyter magic commands (%%writefile, %matplotlib, etc.)
- **Dependency Detection** - Auto-detects required packages (torch, sklearn, pandas, etc.)
- **Endpoint Registry** - Save and reuse deployment endpoints
- **Security Scanning** - Check for hardcoded credentials
- **Metadata Cleaning** - Remove unnecessary notebook metadata

## 🔧 Installation

```bash
# Clone the repository
cd kaggle

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 runna.py doctor
```

### Required Tools
- Python 3.8+
- Kaggle CLI (`pip install kaggle`)
- Google Cloud SDK (for GCP deployments) - [Install Guide](https://cloud.google.com/sdk/docs/install)
- AWS CLI (for AWS deployments) - [Install Guide](https://aws.amazon.com/cli/)
- Modal SDK (for Modal.com) - `pip install modal`

## 🔑 Authentication Setup

### Kaggle API Credentials

**Option 1: Environment Variables** (Recommended for CI/CD)
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Option 2: kaggle.json file**
```bash
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Get your API key from: https://www.kaggle.com/settings/account

### Cloud Provider Authentication

**Google Cloud Platform:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**AWS:**
```bash
aws configure
# Or set environment variables:
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"
```

**Modal.com:**
```bash
modal token new
```

## 📖 Usage Examples

### App Library (Quick Deploy)

**List available apps:**
```bash
python3 runna.py app-list
```

**Deploy pre-built app:**
```bash
# Image classifier with GPU
python3 runna.py app-deploy image-classifier

# Text generator
python3 runna.py app-deploy text-generator --gpu A10G

# Web scraper (no GPU)
python3 runna.py app-deploy web-scraper
```

**Add custom app:**
```bash
python3 runna.py app-add my-app \
  --file my_app.py \
  --description "My custom app" \
  --gpu T4
```

See [APP_LIBRARY.md](APP_LIBRARY.md) for complete app library documentation.

### Send Text to Endpoints

**Send single message:**
```bash
# Deploy LLM chat
python3 runna.py app-deploy llm-chat

# Send message
python3 runna.py send llm-chat "Hello, how are you?"

# Send to URL directly
python3 runna.py send https://your-app.modal.run "Hello"
```

**Interactive chat:**
```bash
python3 runna.py chat llm-chat
```

See [CHAT_GUIDE.md](CHAT_GUIDE.md) for complete chat documentation.

### Basic Notebook Operations

**List your kernels:**
```bash
python3 runna.py list
python3 runna.py list --user username
```

**Pull/Download a notebook:**
```bash
python3 runna.py pull username/kernel-name
python3 runna.py pull username/kernel-name --dest ./notebooks
```

**Push/Upload a notebook:**
```bash
python3 runna.py push ./my-notebook.ipynb
python3 runna.py push ./notebook-directory
```

**Run a notebook on Kaggle:**
```bash
python3 runna.py run username/kernel-name
python3 runna.py run https://kaggle.com/code/user/kernel
python3 runna.py run ./local-notebook.ipynb
python3 runna.py run  # Interactive selection
```

### Deployment Commands

**Deploy to Google Cloud Functions:**
```bash
# Deploy existing notebook directory
python3 runna.py deploy ./notebook-dir \
  --gcp-project my-project \
  --region us-central1 \
  --function-name my-function \
  --memory 1024MB \
  --timeout 540s \
  --save-name my-endpoint

# Run and deploy in one command
python3 runna.py run username/kernel-name --deploy \
  --gcp-project my-project \
  --save-name my-endpoint
```

**Deploy to AWS Lambda:**
```bash
# Package only (creates zip file)
python3 runna.py package-aws ./notebook-dir

# Deploy to AWS Lambda
python3 runna.py deploy-aws ./notebook-dir \
  --function-name my-lambda \
  --role-arn arn:aws:iam::ACCOUNT:role/ROLE_NAME \
  --region us-east-1 \
  --memory 512 \
  --timeout 300 \
  --save-name my-lambda-endpoint
```

**Deploy to Modal.com:**
```bash
# Basic deployment
python3 runna.py deploy-modal ./notebook-dir \
  --save-name my-modal-endpoint

# With GPU support
python3 runna.py deploy-modal ./notebook-dir \
  --gpu A10G \
  --save-name my-gpu-endpoint

# With secrets and custom timeout
python3 runna.py deploy-modal ./notebook-dir \
  --gpu T4 \
  --secrets api-key db-credentials \
  --timeout 600 \
  --save-name production-model

# Available GPU types: T4, A10G, A100
```

**Test Locally:**
```bash
# Package for local testing
python3 runna.py serve-local ./notebook-dir --port 8080

# Run local server
python3 runna.py serve-local ./notebook-dir --run --port 8080
```

### Batch Processing

Create a file `notebooks.txt`:
```
username1/kernel-name-1
username2/kernel-name-2
https://kaggle.com/code/user3/kernel3
```

Process all notebooks:
```bash
# Download all
python3 runna.py batch notebooks.txt --operation download --output-dir ./downloads

# Deploy all
python3 runna.py batch notebooks.txt --operation deploy --output-dir ./deployments
```

### Endpoint Management

**List saved endpoints:**
```bash
python3 runna.py endpoints
```

**Call an endpoint:**
```bash
# Using saved name
python3 runna.py call my-endpoint --json '{"features": [1, 2, 3]}'

# Using direct URL
python3 runna.py call https://my-function-url.com --json '{"data": "test"}'

# From file
python3 runna.py call my-endpoint --json-file payload.json
```

### Notebook Preprocessing

```bash
# Clean metadata and remove outputs
python3 runna.py preprocess ./notebook.ipynb \
  --clean-metadata \
  --remove-outputs \
  --scan-security \
  --output ./cleaned-notebook.ipynb

# Process entire directory
python3 runna.py preprocess ./notebooks/ \
  --clean-metadata \
  --remove-outputs \
  --output ./cleaned-notebooks/
```

## 🛠️ Recent Fixes & Improvements

### ✅ Fixed: Jupyter Magic Command Removal

**Problem:** Notebooks containing Jupyter magic commands (like `%%writefile`, `%matplotlib`, `%load_ext`) would fail to deploy because these commands are not valid Python.

**Solution:** Implemented `clean_jupyter_magic_commands()` function that:
- Removes all cell magic commands (`%%command`)
- Removes all line magic commands (`%command`)
- Filters out `get_ipython()` calls
- Cleans up IPython comment markers (`# In[...]`)
- Preserves all actual Python code

**Example:**
```python
# Before (would fail):
%%writefile task001.py
def my_function():
    pass

# After (works):
def my_function():
    pass
```

### ✅ New: AWS Lambda Deployment

Full support for deploying notebooks to AWS Lambda:
- Automatic packaging as Lambda-compatible zip
- Function creation and updates
- IAM role configuration
- Function URL support
- Environment-based configuration

### ✅ New: Modal.com Deployment

Support for Modal's modern serverless platform:
- Clean Python-first deployment
- Web endpoint creation
- Health check endpoints
- Fast cold starts

### ✅ Improved: Notebook Conversion

Enhanced notebook-to-script conversion:
- Better error handling
- Multiple fallback methods
- Preserves code structure
- Maintains dependencies

## 📝 Notebook Structure for Deployment

### Basic Notebook
```python
# Cell 1: Imports
import pandas as pd
import numpy as np

# Cell 2: Function Definition
def process_request(data):
    """This function will be called by the serverless endpoint"""
    features = data.get('features', [])
    # Your processing logic here
    result = sum(features)
    return {'result': result}

# Cell 3: Optional Test
if __name__ == '__main__':
    test_data = {'features': [1, 2, 3, 4, 5]}
    print(process_request(test_data))
```

### With ML Model
```python
# Cell 1: Imports
import pickle
import numpy as np

# Cell 2: Load Model (will be included in deploy_model.py)
model = pickle.load(open('model.pkl', 'rb'))

# Cell 3: Prediction Function
def process_request(data):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return {'prediction': prediction.tolist()}
```

## 🚨 Common Issues & Solutions

### Issue: "Invalid syntax" error during deployment

**Cause:** Jupyter magic commands in notebook

**Solution:** This is now automatically handled! The tool filters out:
- `%%writefile`, `%%time`, `%%bash`, etc.
- `%matplotlib`, `%load_ext`, etc.
- `get_ipython()` calls

### Issue: "Module not found" in deployed function

**Cause:** Missing dependencies

**Solution:**
1. Create `deploy_model.py` with your dependencies
2. Tool auto-detects: torch, sklearn, pandas, numpy, requests
3. Add custom deps to `requirements.txt` in notebook directory

### Issue: Container healthcheck failed (GCP)

**Cause:** Code tries to import Kaggle-specific paths

**Solution:**
```python
# Add error handling for Kaggle-specific imports
try:
    sys.path.append("/kaggle/input/...")
    from kaggle_module import *
except:
    # Fallback for cloud deployment
    pass
```

### Issue: AWS Lambda deployment fails with "Invalid role ARN"

**Solution:**
```bash
# Create Lambda execution role
aws iam create-role --role-name kaggle-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# Get the ARN
aws iam get-role --role-name kaggle-lambda-role --query 'Role.Arn'

# Use it in deployment
python3 runna.py deploy-aws ./notebook \
  --role-arn arn:aws:iam::123456789:role/kaggle-lambda-role
```

## 🔒 Security Best Practices

1. **Never hardcode credentials** in notebooks
   ```python
   # ❌ Don't do this
   api_key = "sk-abc123..."

   # ✅ Do this
   import os
   api_key = os.environ.get('API_KEY')
   ```

2. **Use security scanning**
   ```bash
   python3 runna.py preprocess notebook.ipynb --scan-security
   ```

3. **Review generated code** before deployment
   ```bash
   # Check the generated main.py
   cat ./notebook-dir/deploy/main.py
   ```

4. **Use private functions** for sensitive data
   ```bash
   python3 runna.py deploy ./notebook --private
   ```

## 📊 Deployment Comparison

| Feature | GCP Functions | AWS Lambda | Modal.com |
|---------|---------------|------------|-----------|
| Cold Start | ~2-5s | ~1-3s | ~1-2s |
| Max Timeout | 60m | 15m | No limit |
| Free Tier | 2M requests/mo | 1M requests/mo | $30/mo credit |
| Python Versions | 3.7-3.11 | 3.8-3.12 | Latest |
| Max Package Size | 500MB | 250MB | 10GB+ |
| GPU Support | ❌ | ❌ | ✅ T4/A10G/A100 |
| Secrets Management | ✅ | ✅ | ✅ |
| Persistent Storage | ❌ | ❌ | ✅ Volumes |
| Scheduled Functions | ✅ | ✅ | ✅ |

### Modal.com Advantages
- **GPU Support**: T4 ($0.60/hr), A10G ($1.10/hr), A100 ($4.00/hr)
- **No Timeout Limits**: Run long-running tasks
- **Large Packages**: Deploy models up to 10GB+
- **Persistent Volumes**: Cache models between invocations
- **Modern Python**: Always latest Python version

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add Azure Functions support
- [ ] Add Vercel/Netlify Functions support
- [ ] Improve error messages
- [ ] Add deployment testing
- [ ] Create Docker deployment option
- [ ] Add monitoring/logging integration

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

- **Issues:** Open an issue on GitHub
- **Documentation:** This README
- **Examples:** See `examples/` directory
- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api

## 📚 Additional Resources

- [Kaggle Notebooks Guide](https://www.kaggle.com/docs/notebooks)
- [GCP Functions Docs](https://cloud.google.com/functions/docs)
- [AWS Lambda Docs](https://docs.aws.amazon.com/lambda/)
- [Modal Docs](https://modal.com/docs)

---

**Version:** 2.0.0
**Last Updated:** 2025-11-02
**Status:** ✅ Production Ready
