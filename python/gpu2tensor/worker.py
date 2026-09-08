"""One worker owns one accelerator and evaluates requests sequentially."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

from gpu2tensor.runner import MAX_REQUEST_BYTES, evaluate_archive


def serve(backend, port, timeout=300):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"backend": backend, "protocol": 2}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/evaluate":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    self.send_error(413)
                    return
                self.connection.settimeout(30)
                payload = self.rfile.read(size)
                if len(payload) != size:
                    raise ValueError("Incomplete request body.")
                body = evaluate_archive(payload, backend, timeout)
            except Exception as error:
                self.send_error(400, explain=str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    # This executes trusted source. Only expose it through operator-owned tunnels.
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["cuda", "trainium", "cpu"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    serve(args.backend, args.port, args.timeout)


if __name__ == "__main__":
    main()
