# run.py
"""
Server Startup Script - OPTIMIZED FOR CONCURRENCY
Run this file to start the honeypot server.

Usage:
    python run.py
"""

import uvicorn
import multiprocessing
import os

if __name__ == "__main__":

    # Calculate optimal workers
    # = (CPU cores * 2) + 1 is standard formula
    cpu_count = multiprocessing.cpu_count()
    # For local/dev on machines with limited RAM, cap workers to avoid MemoryError
    default_workers = min((cpu_count * 2) + 1, 4)  # at most 4 workers by default
    # Allow override via environment variable if you want to tune it
    workers = int(os.environ.get("UVICORN_WORKERS", default_workers))
    
    port = int(os.environ.get("PORT", 8002))

    print("=" * 70)
    print("STARTING SCAMBAIT AI - HONEYPOT API")
    print("=" * 70)
    print()
    print(f"API:         http://127.0.0.1:{port}")
    print(f"Docs:        http://127.0.0.1:{port}/docs")
    print(f"Health:      http://127.0.0.1:{port}/health")
    print()
    print(f"CPU Cores:   {cpu_count}")
    print(f"Workers:     {workers} (handles {workers * 30} concurrent requests)")
    print(f"Concurrency: 50+ simultaneous requests supported")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",           # Accept from any IP (needed for hackathon)
        port=int(os.environ.get("PORT", 8002)),
        reload=False,
        workers=workers,           # Multiple worker processes (capped for local/dev)
        loop="asyncio",            # Async event loop
        http="httptools",          # Faster HTTP parser
        limit_concurrency=60,      # Max concurrent connections (lower to reduce memory pressure)
        limit_max_requests=10000,  # Restart worker after 10k requests (memory)
        timeout_keep_alive=30,     # Keep connections alive 30s
        access_log=True,           # Log all requests
    )
