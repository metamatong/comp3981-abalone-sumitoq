"""Abalone web server — serves the HTML UI and JSON API."""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from .session import GameSession

session = GameSession()

PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(PACKAGE_DIR, "static")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html")
        elif self.path == "/style.css":
            self._serve_file("style.css", "text/css")
        elif self.path == "/script.js":
            self._serve_file("script.js", "application/javascript")
        elif self.path == "/api/state":
            self._json_response(session.state_json())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON payload"})
            return

        if self.path == "/api/move":
            self._json_response(session.apply_human_move(body))
        elif self.path == "/api/agent-move":
            self._json_response(session.apply_agent_move())
        elif self.path == "/api/undo":
            self._json_response(session.undo())
        elif self.path == "/api/reset":
            self._json_response(session.reset())
        elif self.path == "/api/pause":
            self._json_response(session.toggle_pause())
        elif self.path == "/api/resign":
            self._json_response(session.resign())
        elif self.path == "/api/config":
            self._json_response(session.configure(body))
        else:
            self.send_error(404)

    def _json_response(self, data):
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(payload))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, name: str, mime: str):
        path = os.path.join(STATIC_DIR, name)
        with open(path, "rb") as file:
            data = file.read()

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def run(port: int = 9000):
    import socket
    import webbrowser

    for p in [port] + list(range(port + 1, port + 20)):
        try:
            sock = socket.socket()
            sock.bind(("", p))
            sock.close()
            port = p
            break
        except OSError:
            continue

    server = HTTPServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"Abalone running at  {url}")
    webbrowser.open(url)
    server.serve_forever()
