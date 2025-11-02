import os
import sys
import subprocess

def check_python_version():
    print("--- Checking Python Version ---")
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        return False
    print("Success: Python version is compatible.")
    return True

def check_kaggle_package():
    print("\n--- Checking Kaggle Package ---")
    try:
        import kaggle
        print("Success: Kaggle package is installed.")
        return True
    except ImportError:
        print("Error: Kaggle package is not installed.")
        print("Please run: pip install kaggle")
        return False

def check_kaggle_credentials():
    print("\n--- Checking Kaggle Credentials ---")
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if os.path.exists(kaggle_json_path):
        print(f"Success: Found kaggle.json at {kaggle_json_path}")
        return True
    elif kaggle_username and kaggle_key:
        print("Success: Found KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return True
    else:
        print("Error: Could not find Kaggle credentials.")
        print("Please either place your kaggle.json file in ~/.kaggle/ or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return False

def check_kaggle_api():
    print("\n--- Checking Kaggle API Connection ---")
    try:
        result = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, check=True)
        if "expiring-access-token" in result.stdout:
            print("Success: Successfully connected to the Kaggle API.")
            return True
        else:
            print("Error: Could not connect to the Kaggle API. Please check your credentials.")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not run the kaggle command. Please make sure it is installed and in your PATH.")
        return False

if __name__ == "__main__":
    print("Running Kaggle Environment Doctor...")
    results = [
        check_python_version(),
        check_kaggle_package(),
        check_kaggle_credentials(),
        check_kaggle_api(),
    ]

    if all(results):
        print("\nYour environment seems to be set up correctly!")
    else:
        print("\nPlease fix the errors above and try again.")
