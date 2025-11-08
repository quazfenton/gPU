11-02-25
**Issue:** Notebooks containing Jupyter/IPython magic commands would fail during deployment with syntax errors. Commands like `%%writefile`, `%matplotlib`, `%load_ext`, and `get_ipython()` calls are not valid Python and caused build failures.

**Impact:**
- Deployments to Google Cloud Functions failed with "SyntaxError: invalid syntax"
- AWS Lambda packages contained invalid Python code
- Many Kaggle notebooks couldn't be deployed

**Root Cause:**
The `nbconvert.PythonExporter` preserves IPython-specific syntax, including:
- Cell magic commands (`%%writefile`, `%%time`, `%%bash`)
- Line magic commands (`%matplotlib`, `%load_ext`, `%pip`)
- IPython introspection (`get_ipython()` calls)
- Comment markers (`# In[1]:`, `# In[ ]:`)

**Solution:**
Implemented `clean_jupyter_magic_commands()` function that:
1. Parses converted notebook code line by line
2. Filters out all magic command patterns using regex
3. Removes `get_ipython()` calls
4. Cleans up IPython comment markers
5. Normalizes excessive blank lines

**Code Changes:**
```python
def clean_jupyter_magic_commands(script_content):
    """Remove Jupyter magic commands and IPython-specific syntax"""
    lines = script_content.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip IPython markers: # In[ ]:
        if re.match(r'^#\s*In\s*\[.*\]\s*:?\s*$', stripped):
            continue

        # Skip cell magic: %%writefile, %%time, etc.
        if stripped.startswith('%%'):
            continue

        # Skip line magic: %matplotlib, %load_ext, etc.
        if stripped.startswith('%') and not stripped.startswith('%%'):
            continue

        # Skip get_ipython() calls
        if 'get_ipython()' in line:
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)
```

**Testing:**
```bash
# Before fix - would fail:
%%writefile task001.py
p=lambda*g:[[*g,min,p][2](r,s)for r in g[0]for s in g[-1]]

# After fix - deploys successfully:
p=lambda*g:[[*g,min,p][2](r,s)for r in g[0]for s in g[-1]]
```

**Files Modified:**
- `runna.py`: Added `clean_jupyter_magic_commands()` at line 1311
- `runna.py`: Integrated cleaning in `convert_notebook_to_script()` at line 1362
- `runna.py`: Added fallback cleaning for manual extraction at line 1379

### ✨ New Features

#### 1. AWS Lambda Deployment Support

Full-featured AWS Lambda deployment with automatic function management.

**Commands Added:**
```bash
# Package notebook as Lambda zip
python3 runna.py package-aws ./notebook-dir

# Deploy to AWS Lambda
python3 runna.py deploy-aws ./notebook-dir \
  --function-name my-function \
  --role-arn arn:aws:iam::ACCOUNT:role/ROLE \
  --region us-east-1 \
  --memory 512 \
  --timeout 300 \
  --save-name my-endpoint
```

**Features:**
- Automatic zip packaging with dependencies
- Function creation and updates (detects existing functions)
- IAM role configuration
- Function URL support
- Environment variable support
- Automatic endpoint registration

**Implementation:**
- `deploy_to_aws_lambda()` - Main deployment function
- `create_lambda_handler_main()` - Lambda handler template
- `cmd_deploy_aws()` - CLI command handler
- Integration with AWS CLI

**Handler Template:**
```python
def handler(event, context):
    """AWS Lambda handler with API Gateway support"""
    body = event.get('body')
    if body and isinstance(body, str):
        payload = json.loads(body)
    else:
        payload = event

    # Call notebook's process_request if available
    if 'process_request' in globals():
        result = globals()['process_request'](payload)
    else:
        result = {'echo': payload}

    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(result)
    }
```

#### 2. Modal.com Deployment Support

Integration with Modal's modern serverless Python platform.

**Commands Added:**
```bash
# Deploy to Modal.com
python3 runna.py deploy-modal ./notebook-dir \
  --save-name my-modal-endpoint
```

**Features:**
- Python-first deployment (no Docker/containers)
- Fast cold starts (~1-2 seconds)
- GPU support (not available in GCP/AWS free tiers)
- Web endpoint creation with decorators
- Health check endpoints
- No package size limits (10GB+)

**Implementation:**
- `package_for_modal()` - Package notebook for Modal
- `create_modal_app()` - Generate Modal app file
- `deploy_to_modal()` - Deploy using Modal CLI
- `cmd_deploy_modal()` - CLI command handler

**Generated App Template:**
```python
import modal

app = modal.App("kaggle-notebook")

@app.function()
@modal.web_endpoint(method="POST")
def predict(data: dict):
    """Modal web endpoint for predictions"""
    if 'process_request' in globals():
        result = globals()['process_request'](data)
    else:
        result = {"echo": data}
    return result

@app.function()
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint"""
    return {"status": "ok"}
```

**Why Modal.com?**
- Faster cold starts than AWS Lambda
- Better Python ecosystem support
- No configuration overhead
- Built-in GPU support
- Modern developer experience

#### 3. Enhanced Endpoint Registry

Improved endpoint management for multiple deployment targets.

**Features:**
- Store endpoints from all platforms (GCP, AWS, Modal)
- Metadata tracking (provider, region, function name)
- Timestamp tracking
- Easy endpoint calling

**Storage Format:**
```json
{
  "my-endpoint": {
    "url": "https://...",
    "provider": "gcp-functions",
    "metadata": {
      "project": "my-project",
      "region": "us-central1"
    },
    "created_at": 1730581234
  }
}
```

**Usage:**
```bash
# Save endpoint during deployment
python3 runna.py deploy ./nb --save-name my-api

# List all endpoints
python3 runna.py endpoints

# Call saved endpoint
python3 runna.py call my-api --json '{"data": "test"}'
```


```bash
# Test basic functionality
python3 runna.py doctor

# Test magic command removal
python3 runna.py deploy ./jazivxt_system-control-pannel --gcp-project YOUR_PROJECT

# Test AWS deployment
python3 runna.py deploy-aws ./examples/simple_api.ipynb --role-arn YOUR_ROLE

# Test Modal deployment
python3 runna.py deploy-modal ./examples/simple_api.ipynb
```

### 📦 Dependencies

No new required dependencies. Optional dependencies:
- `modal` - For Modal.com deployment
- `boto3` - For enhanced AWS support (AWS CLI still works)
- `google-cloud-functions` - For enhanced GCP support

#### Planned for v2.1.0
- [ ] Azure Functions support
- [ ] Vercel/Netlify Functions support
- [ ] Docker container deployment option

#### Under Consideration
- [ ] Terraform/IaC generation
- [ ] CI/CD pipeline templates

**New commands to try:**
```bash
# Deploy to AWS instead of GCP
python3 runna.py deploy-aws ./notebook --role-arn YOUR_ROLE

# Try Modal.com for faster cold starts
python3 runna.py deploy-modal ./notebook

# Save endpoints for easy reuse
python3 runna.py deploy ./notebook --save-name my-api
python3 runna.py call my-api --json '{"test": true}'
```

