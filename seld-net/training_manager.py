import subprocess
import sys

def main():
    """
    Manages the SELDnet training loop by repeatedly calling seld_continue.py.

    This script replaces the run_training_loop.sh and provides a pure Python
    solution to run training in sessions, which helps prevent memory-related
    crashes during long training runs.
    """
    if len(sys.argv) != 3:
        print("\nUsage: python training_manager.py <job-id> <task-id>")
        print("Example: python training_manager.py 59_50LossWeights 22\n")
        sys.exit(1)

    job_id = sys.argv[1]
    task_id = sys.argv[2]

    # Use the same Python interpreter that is running this manager script
    # to ensure the same environment (e.g., your '(tf)' venv) is used.
    python_executable = sys.executable

    # The script to run, assumed to be in the same directory as this manager.
    script_to_run = 'seld-net/seld_continue.py'

    print(f"Starting continuous training for job-id: {job_id}, task-id: {task_id}")
    print(f"Using Python interpreter: {python_executable}")

    while True:
        print("\n" + "-"*50)
        print("--- Starting a new training session...           ---")
        print("-"*50)

        command = [python_executable, script_to_run, job_id, task_id]

        # Run the training script as a subprocess and wait for it to complete.
        process = subprocess.run(command)
        exit_code = process.returncode

        if exit_code == 100:
            print("\n" + "-"*50, "\n--- Training completed successfully.           ---\n" + "-"*50)
            break
        elif exit_code != 0:
            print("\n" + "-"*50, f"\n--- Training script failed with exit code {exit_code}. ---\n--- Aborting loop.                           ---\n" + "-"*50)
            break

if __name__ == "__main__":
    main()