#!/bin/bash

# This script runs the main CLI agent Python script.

# Ensure the virtual environment is activated before running this script.
# Example: source .venv/bin/activate

echo "Running the CLI Agent..."
python -m src.main

# Check the exit code of the Python script
exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo "Agent script exited with error code $exit_code."
  exit $exit_code
else
  echo "Agent script finished successfully."
fi

exit 0 