#!/usr/bin/env python
"""
Quick TensorBoard launcher using the tensorboard.main API.
Avoids entry point issues on Windows.
"""
import sys
import os

# Add conda env Scripts to PATH to avoid module not found
os.environ['PATH'] = r'C:\Users\student\.conda\envs\gpu-jit-opt\Scripts' + os.pathsep + os.environ['PATH']

try:
    # Import and run TensorBoard main
    from tensorboard.main import run
    
    # Arguments: logdir and port
    sys.argv = [
        'tensorboard',
        '--logdir', 'results/logs/tensorboard',
        '--port', '6006',
        '--host', '0.0.0.0'
    ]
    
    print("Launching TensorBoard on http://localhost:6006")
    print("Press Ctrl+C to stop.")
    run()
    
except ImportError as e:
    print(f"Error importing TensorBoard: {e}")
    print("Trying fallback method...")
    
    # Fallback: direct subprocess call
    import subprocess
    subprocess.run([
        sys.executable, '-m', 'tensorboard.main',
        '--logdir', 'results/logs/tensorboard',
        '--port', '6006'
    ])
