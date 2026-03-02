#Clean django project entrypoint. For local development only.
# Minimal entrypoint — for local development only.
# For production, use: gunicorn pfp_validator.wsgi
import subprocess
import sys

print("Starting Django development server...")

if __name__ == "__main__":
    subprocess.run([
        "uv", "run", "python", "manage.py", "runserver", "127.0.0.1:8000"
    ])
