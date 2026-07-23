from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
print(f"Starting server on port {sys.argv[1] if len(sys.argv) > 1 else 8000}", flush=True)
HTTPServer(('127.0.0.1', int(sys.argv[1]) if len(sys.argv) > 1 else 8000), SimpleHTTPRequestHandler).serve_forever()
