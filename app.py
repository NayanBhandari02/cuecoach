import subprocess
import sys

processes = []

try:
    backend = subprocess.Popen(
        ["uvicorn", "src.cuecoach.api.main:app", "--reload"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    frontend = subprocess.Popen(
        ["streamlit", "run", "streamlit_app.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    processes.append(backend)
    processes.append(frontend)

    for p in processes:
        p.wait()

except KeyboardInterrupt:
    for p in processes:
        p.terminate()