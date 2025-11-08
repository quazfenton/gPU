"""App library manager for Modal serverless functions."""
import json
from pathlib import Path
from datetime import datetime

APPS_DIR = Path(__file__).parent / "apps"
LIBRARY_INDEX = APPS_DIR / "library.json"

def load_library():
    """Load app library index."""
    if LIBRARY_INDEX.exists():
        return json.loads(LIBRARY_INDEX.read_text())
    return {"apps": {}}

def save_library(data):
    """Save app library index."""
    APPS_DIR.mkdir(exist_ok=True)
    LIBRARY_INDEX.write_text(json.dumps(data, indent=2))

def list_apps():
    """List all apps in library."""
    lib = load_library()
    return lib.get("apps", {})

def add_app(name, code, description="", tags=None, gpu=None):
    """Add app to library."""
    lib = load_library()
    app_file = APPS_DIR / f"{name}.py"
    app_file.write_text(code)
    
    lib["apps"][name] = {
        "file": f"{name}.py",
        "description": description,
        "tags": tags or [],
        "gpu": gpu,
        "created": datetime.now().isoformat(),
        "deployed": None
    }
    save_library(lib)
    return app_file

def get_app(name):
    """Get app code."""
    lib = load_library()
    if name not in lib["apps"]:
        return None
    app_file = APPS_DIR / lib["apps"][name]["file"]
    return app_file.read_text() if app_file.exists() else None

def update_app(name, code=None, deployed_url=None):
    """Update app code or deployment info."""
    lib = load_library()
    if name not in lib["apps"]:
        return False
    
    if code:
        app_file = APPS_DIR / lib["apps"][name]["file"]
        app_file.write_text(code)
    
    if deployed_url:
        lib["apps"][name]["deployed"] = deployed_url
        lib["apps"][name]["last_deploy"] = datetime.now().isoformat()
    
    save_library(lib)
    return True

def delete_app(name):
    """Delete app from library."""
    lib = load_library()
    if name not in lib["apps"]:
        return False
    
    app_file = APPS_DIR / lib["apps"][name]["file"]
    if app_file.exists():
        app_file.unlink()
    
    del lib["apps"][name]
    save_library(lib)
    return True
