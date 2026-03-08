# -*- coding: utf-8 -*-
"""
MD AI Analyzer — Entry Point
Run this to start the local web server.
"""
import uvicorn
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import HOST, PORT, GPU_AVAILABLE, GPU_NAME


def main():
    print("=" * 60)
    print("  🧬 MD AI Analyzer")
    print("  AI-Powered Molecular Dynamics Analysis Platform")
    print("=" * 60)
    print()
    if GPU_AVAILABLE:
        print(f"  ⚡ GPU Detected: {GPU_NAME}")
    else:
        print("  💻 Running on CPU (GPU not detected)")
    print(f"  🌐 Server: http://localhost:{PORT}")
    print(f"  📁 Upload directory: uploads/")
    print(f"  📊 Results directory: results/")
    print()
    print("  Open http://localhost:8000 in your browser to begin.")
    print("=" * 60)

    uvicorn.run(
        "backend.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
