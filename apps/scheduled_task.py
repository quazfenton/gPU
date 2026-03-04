"""Scheduled task example - Production Ready."""
import modal
import requests
from datetime import datetime
import json

app = modal.App("scheduled-task")
image = modal.Image.debian_slim().pip_install("requests")

# Create volume for task logs
task_logs_volume = modal.Volume.from_name("task-logs", create_if_missing=True)

@app.function(
    image=image,
    schedule=modal.Cron("0 * * * *"),  # Runs every hour
    volumes={"/logs": task_logs_volume},
    timeout=300,
)
def hourly_task():
    """Runs every hour with logging and error handling."""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = f"/logs/{task_id}.json"

    result = {
        "task_id": task_id,
        "scheduled_time": datetime.now().isoformat(),
        "status": "unknown",
        "output": None,
        "error": None,
    }

    try:
        print(f"Starting task {task_id} at {datetime.now()}")
        result["status"] = "running"

        # Your task logic here
        # Example: Fetch data, process, store results
        output = {
            "message": "Task completed successfully",
            "timestamp": datetime.now().isoformat(),
        }

        result["status"] = "completed"
        result["output"] = output
        print(f"Task {task_id} completed successfully")

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"Task {task_id} failed: {e}")

    finally:
        # Always save result
        try:
            with open(log_file, 'w') as f:
                json.dump(result, f, indent=2)
            task_logs_volume.commit()
        except Exception as save_error:
            print(f"Failed to save task result: {save_error}")

    return result

@app.function(image=image)
@modal.web_endpoint(method="GET")
def status():
    """Check task status and recent executions."""
    try:
        # List recent task logs
        recent_tasks = []
        try:
            for entry in task_logs_volume.listdir("/"):
                if entry.name.endswith('.json'):
                    recent_tasks.append(entry.name)
        except Exception:
            pass

        return {
            "status": "active",
            "schedule": "hourly (0 * * * *)",
            "recent_executions": sorted(recent_tasks)[-10:],
            "next_scheduled": "top of next hour",
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.function(image=image)
@modal.web_endpoint(method="GET")
def get_task_result(task_id: str):
    """Get result of a specific task execution."""
    try:
        log_file = f"/logs/{task_id}.json"
        with open(log_file, 'r') as f:
            result = json.load(f)
        return result
    except FileNotFoundError:
        return {"error": f"Task {task_id} not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500
