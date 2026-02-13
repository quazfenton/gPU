# Quick Start Guide

Get started with Kaggle Notebook Automation in 5 minutes!

## ⚡ Quick Install

```bash
cd kaggle
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 Set Up Credentials

**Get your Kaggle API key:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json`

**Configure:**
```bash
# Option 1: File (easiest)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Option 2: Environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

**Verify:**
```bash
python3 runna.py doctor
```

## 🎯 First Commands

### Download a Notebook
```bash
# From Kaggle URL
python3 runna.py pull username/kernel-name

# Interactive selection
python3 runna.py run
```

### Deploy Your First API

**1. Use the example notebook:**
```bash
cd examples
python3 ../runna.py serve-local simple_api.ipynb --run --port 8080
```

**2. Test it:**
```bash
# In another terminal
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{"operation": "calculate", "numbers": [1,2,3,4,5]}'
```

Expected output:
```json
{
  "status": "success",
  "result": {
    "sum": 15,
    "mean": 3.0,
    "max": 5,
    "min": 1,
    "count": 5
  }
}
```

## ☁️ Deploy to Cloud

### Google Cloud Functions

**Setup:**
```bash
# Install gcloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**Deploy:**
```bash
python3 runna.py deploy examples/simple_api.ipynb \
  --gcp-project YOUR_PROJECT_ID \
  --save-name my-first-api
```

**Call it:**
```bash
python3 runna.py call my-first-api --json '{"operation": "info"}'
```

### AWS Lambda

**Setup:**
```bash
# Install AWS CLI: https://aws.amazon.com/cli/
aws configure
```

**Create IAM Role (one-time):**
```bash
# Create trust policy
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

# Create role
aws iam create-role --role-name kaggle-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# Attach basic execution policy
aws iam attach-role-policy --role-name kaggle-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Get the ARN (copy this!)
aws iam get-role --role-name kaggle-lambda-role --query 'Role.Arn' --output text
```

**Deploy:**
```bash
python3 runna.py deploy-aws examples/simple_api.ipynb \
  --role-arn arn:aws:iam::YOUR_ACCOUNT:role/kaggle-lambda-role \
  --save-name my-aws-api
```

### Modal.com (Fastest!)

**Setup:**
```bash
pip install modal
modal token new  # Opens browser for auth
```

**Deploy:**
```bash
python3 runna.py deploy-modal examples/simple_api.ipynb \
  --save-name my-modal-api
```

## 📋 Common Tasks

### Work with Your Own Notebooks

**Create deployable notebook:**
```python
# Your notebook should have this function:
def process_request(data):
    """Your API logic here"""
    result = do_something(data)
    return {"result": result}
```

**Deploy it:**
```bash
python3 runna.py deploy your_notebook.ipynb --gcp-project YOUR_PROJECT
```

### Batch Download
```bash
# Create notebooks.txt
echo "username1/kernel1" > notebooks.txt
echo "username2/kernel2" >> notebooks.txt

# Download all
python3 runna.py batch notebooks.txt --operation download
```

### Clean & Secure Notebooks
```bash
python3 runna.py preprocess notebook.ipynb \
  --clean-metadata \
  --remove-outputs \
  --scan-security \
  --output clean_notebook.ipynb
```

## 🔧 Troubleshooting

### "Kaggle API credentials not found"
→ Run `python3 runna.py doctor` to diagnose
→ Ensure kaggle.json is in ~/.kaggle/ or set env vars

### "gcloud command not found"
→ Install: https://cloud.google.com/sdk/docs/install
→ Or use AWS/Modal instead

### "SyntaxError" during deployment
→ This is now automatically fixed! Update to latest version.

### Notebook has Kaggle-specific imports
Add error handling:
```python
try:
    sys.path.append("/kaggle/input/...")
    from kaggle_module import *
except:
    print("Running outside Kaggle, skipping...")
```

## 💡 Pro Tips

1. **Save endpoints** for easy reuse:
   ```bash
   python3 runna.py deploy ./nb --save-name prod-api
   python3 runna.py call prod-api --json '{"test": true}'
   ```

2. **Test locally first:**
   ```bash
   python3 runna.py serve-local ./nb --run --port 8080
   ```

3. **Use environment variables** for secrets:
   ```python
   import os
   api_key = os.environ.get('API_KEY')  # Not hardcoded!
   ```

4. **Check deployment costs:**
   - GCP: 2M free requests/month
   - AWS: 1M free requests/month
   - Modal: $30 free credit/month

## 📚 Next Steps

- Read [README.md](README.md) for complete documentation
- Check [CHANGELOG.md](CHANGELOG.md) for latest features
- Browse [examples/](examples/) for more samples
- Run `python3 runna.py COMMAND --help` for command details

## 🆘 Quick Help

```bash
# List all commands
python3 runna.py --help

# Help for specific command
python3 runna.py deploy --help

# Check environment
python3 runna.py doctor

# List saved endpoints
python3 runna.py endpoints
```

## ✅ Checklist

- [ ] Install dependencies
- [ ] Configure Kaggle credentials
- [ ] Run `runna.py doctor`
- [ ] Test local server with example
- [ ] Deploy to one cloud platform
- [ ] Call your deployed API
- [ ] Save endpoint for reuse

**You're ready to go! 🚀**

For questions, check the full [README.md](README.md) or open an issue.
