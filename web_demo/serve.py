from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    
    def do_GET(self):
        print(f"Serving: {self.path}")
        super().do_GET()

socketserver.TCPServer.allow_reuse_address = True
server = socketserver.TCPServer(('127.0.0.1', 8000), Handler)
print("Serving on http://127.0.0.1:8000 with COOP/COEP headers")
server.serve_forever()
