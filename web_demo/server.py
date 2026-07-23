#!/usr/bin/env python3
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class COOPHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
port = int(os.sys.argv[1]) if len(os.sys.argv) > 1 else 8000
print(f"Serving on port {port} with COOP/COEP headers")
HTTPServer(('127.0.0.1', port), COOPHandler).serve_forever()
