"""Sahte llama-server: `--port` dinler, `/health` 200 döner (testlerde exe yerine python ile çalıştırılır)."""
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    delay = float(args[args.index("--delay") + 1]) if "--delay" in args else 0.0
    ngl = int(args[args.index("-ngl") + 1]) if "-ngl" in args else 0
    fail_above = int(args[args.index("--fail-above") + 1]) if "--fail-above" in args else None
    if fail_above is not None and ngl > fail_above:
        sys.exit(3)  # OOM benzetimi
    if "--invalid" in args:
        print("error: invalid argument: --foo", flush=True)
        sys.exit(1)
    time.sleep(delay)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, *a):
            pass

    HTTPServer(("127.0.0.1", port), H).serve_forever()


main()
