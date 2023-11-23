import subprocess
import time
import signal
import os

def run_with_timeout(command, timeout):
    """Run command with the specified timeout. Kill it if it doesn't finish within the timeout."""

    # Start the subprocess
    proc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

    try:
        # Wait for process to complete or timeout to expire
        proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Timeout expired. Kill the process group.
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.communicate()  # Ensure it's terminated before moving on
    except KeyboardInterrupt:
        # User pressed Ctrl+C. Kill the process group and re-raise the exception
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.communicate()
        raise

    return

def main():
    total_duration = 8 * 60 * 60  # 8 hours in seconds
    script_duration = 20 * 60  # 20 minutes in seconds
    command = "python3 'pre-processing/create_mesh.py'"  # Assuming Python3, adjust accordingly

    start_time = time.time()

    try:
        while time.time() - start_time < total_duration:
            run_with_timeout(command, script_duration)
            # Consider adding a sleep here if needed to avoid immediate restart
    except KeyboardInterrupt:
        print("\nScheduler interrupted. Exiting gracefully.")

if __name__ == "__main__":
    main()
