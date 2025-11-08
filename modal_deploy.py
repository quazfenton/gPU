"""
Enhanced Modal.com deployment for Kaggle notebooks.
Supports GPU, secrets, volumes, and scheduled functions.
"""
import modal

app = modal.App("kaggle-notebook")

# Define image with dependencies
image = modal.Image.debian_slim().pip_install(
    "numpy", "pandas", "scikit-learn", "torch", "requests"
)

# Optional: Mount volumes for model storage
volume = modal.Volume.from_name("model-cache", create_if_missing=True)

@app.function(
    image=image,
    gpu=None,  # Set to "T4", "A10G", "A100" for GPU
    secrets=[],  # Add modal.Secret.from_name("my-secret")
    volumes={"/cache": volume},
    timeout=300,
)
@modal.web_endpoint(method="POST")
def predict(data: dict):
    """Main prediction endpoint."""
    try:
        # Import notebook code
        from notebook_code import process_request
        return process_request(data)
    except ImportError:
        return {"error": "process_request not found"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.function(
    image=image,
    schedule=modal.Cron("0 0 * * *"),  # Daily at midnight
)
def scheduled_task():
    """Optional scheduled task."""
    print("Running scheduled task")
