"""Scheduled task example."""
import modal

app = modal.App("scheduled-task")
image = modal.Image.debian_slim().pip_install("requests")

@app.function(image=image, schedule=modal.Cron("0 * * * *"))
def hourly_task():
    """Runs every hour."""
    import requests
    from datetime import datetime
    
    print(f"Task running at {datetime.now()}")
    # Your task logic here
    return {"status": "completed", "time": datetime.now().isoformat()}

@app.function(image=image)
@modal.web_endpoint(method="GET")
def status():
    """Check task status."""
    return {"status": "active", "schedule": "hourly"}
