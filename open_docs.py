#!/usr/bin/env python3
"""Quick launcher for project documentation.

Opens docs/index.html in the default web browser.

Usage:
    python open_docs.py
    
    or on Windows:
    
    python open_docs.py --server  # Launches local http server
"""

import sys
import webbrowser
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Open project documentation")
    parser.add_argument("--server", action="store_true",
                        help="Launch local HTTP server instead of opening file directly")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for HTTP server (default: 8000)")
    
    args = parser.parse_args()
    
    # Get paths
    script_dir = Path(__file__).resolve().parent
    docs_dir = script_dir / "docs"
    docs_file = docs_dir / "index.html"
    
    if not docs_file.exists():
        print(f"Error: Documentation file not found at {docs_file}")
        print(f"Expected location: {docs_dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Let It Compile: RL Adaptive Register Allocation for GPU Kernels - Documentation")
    print("=" * 70)
    print()
    
    if args.server:
        print(f"Starting local HTTP server on port {args.port}...")
        print()
        print(f"Documentation will be available at:")
        print(f"    http://localhost:{args.port}")
        print()
        print("Press Ctrl+C to stop the server.")
        print()
        
        # Change to docs directory and start server
        os.chdir(docs_dir)
        
        try:
            import http.server
            import socketserver
            
            Handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer(("", args.port), Handler) as httpd:
                webbrowser.open(f"http://localhost:{args.port}")
                print(f"Server started. Browser should open automatically...")
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
            sys.exit(0)
        except OSError as e:
            print(f"Error: Could not start server on port {args.port}")
            print(f"Reason: {e}")
            print(f"\nTry a different port: python open_docs.py --server --port 8001")
            sys.exit(1)
    else:
        # Open file directly
        file_url = docs_file.as_uri()
        print(f"Opening documentation...")
        print(f"File: {docs_file}")
        print()
        
        try:
            webbrowser.open(file_url)
            print("Documentation opened in your default browser.")
        except Exception as e:
            print(f"Error opening browser: {e}")
            print(f"Please open this file manually: {docs_file}")
            sys.exit(1)

if __name__ == "__main__":
    main()
