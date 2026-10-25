"""A small mock of an insurer's policy administration API, serving the sample records.

    python -m clauselens.policies.api              # http://localhost:8600
    python -m clauselens.policies.api --port 8700

Endpoints:
    GET /health                    {"status": "ok"}
    GET /policies                  list of policy records
    GET /policies/{policy_number}  one record, or 404

Then set CL_POLICY_SOURCE=http (and CL_POLICY_API_URL if not the default) so ClauseLens
fetches records through this API instead of reading the file directly.
Standard library only; for local demos, not production.
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from clauselens.config import get_settings
from clauselens.policies.records import FilePolicySystem, PolicySystem


def make_handler(source: PolicySystem) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            path = urllib.parse.urlparse(self.path).path.rstrip("/")
            if path == "/health":
                self._send(200, {"status": "ok"})
            elif path == "/policies":
                self._send(200, [p.to_dict() for p in source.list_policies()])
            elif path.startswith("/policies/"):
                number = urllib.parse.unquote(path.removeprefix("/policies/"))
                record = source.get_policy(number)
                if record:
                    self._send(200, record.to_dict())
                else:
                    self._send(404, {"error": f"policy {number} not found"})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, fmt, *args) -> None:
            print(f"[policy-api] {self.address_string()} {fmt % args}")

    return Handler


def serve(port: int = 8600, source: PolicySystem | None = None) -> ThreadingHTTPServer:
    source = source or FilePolicySystem(get_settings().policies_file)
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    server = serve(args.port)
    print(f"Sample policy API running at http://localhost:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
