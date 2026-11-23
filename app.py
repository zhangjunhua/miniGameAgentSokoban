"""Run the local Sokoban laboratory with: python app.py"""

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from game_session import RequestError, RunStore
from model_client import ModelClient, load_local_keys, load_models
from model_log import ModelCallLog
from sokoban import LEVELS

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
RUN_ACTION = re.compile(r"/api/runs/([0-9a-f]{32})/(step|end)")


class SokobanServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, models, client=None):
        super().__init__(address, Handler)
        self.runs = RunStore(models)
        self.client = client if client is not None else ModelClient(call_log=ModelCallLog(ROOT / "log"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self.handle_request("GET")

    def do_POST(self):
        self.handle_request("POST")

    def handle_request(self, method):
        try:
            self.check_local_origin()
            if method == "GET" and self.path in ASSETS:
                filename, content_type = ASSETS[self.path]
                self.send_bytes((ROOT / "web" / filename).read_bytes(), content_type)
                return
            payload = self.get_api() if method == "GET" else self.post_api()
            self.send_json(payload)
        except RequestError as error:
            self.send_json({"error": str(error)}, error.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as error:
            self.log_error("Sokoban request failed (%s)", type(error).__name__)
            self.send_json({"error": "本地服务处理失败，请重开本局。"}, 500)

    def check_local_origin(self):
        port = self.server.server_port
        host = self.headers.get("Host", "").lower()
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update(("127.0.0.1", "localhost"))
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin.lower() != f"http://{host}"):
            raise RequestError("请从本机游戏地址访问。", 403)

    def get_api(self):
        if self.path == "/api/levels":
            return {"levels": [
                {"id": level.id, "name": level.name, "description": level.description}
                for level in LEVELS.values()
            ]}
        if self.path == "/api/models":
            return {"models": [model.public() for model in self.server.runs.models.values()]}
        raise RequestError("未找到这个页面。", 404)

    def post_api(self):
        if self.path == "/api/runs":
            return self.server.runs.create(self.read_json())
        match = RUN_ACTION.fullmatch(self.path)
        if not match:
            raise RequestError("未找到这个接口。", 404)
        run_id, action = match.groups()
        request = self.read_json()
        run = self.server.runs.get(run_id)
        if action == "end":
            return {"run": run.end()}
        return run.step(request, self.server.client)

    def read_json(self):
        if self.headers.get_content_type() != "application/json":
            raise RequestError("请发送 JSON 请求。", 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0:
                raise ValueError()
            if size > 1_100_000:
                raise RequestError("请求过大。", 413)
            payload = json.loads(self.rfile.read(size))
        except (ValueError, RecursionError):
            raise RequestError("JSON 请求无效。") from None
        if not isinstance(payload, dict):
            raise RequestError("JSON 请求必须是对象。")
        return payload

    def send_json(self, payload, status=200):
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json", status)

    def send_bytes(self, content, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    parser = argparse.ArgumentParser(description="推箱子实验室")
    parser.add_argument("--port", type=int, default=8765, help="本地网页端口")
    args = parser.parse_args()
    try:
        load_local_keys(ROOT / ".env")
    except (OSError, UnicodeError):
        print("无法读取项目根目录的 .env。请检查文件权限和 UTF-8 编码。", file=sys.stderr)
    try:
        models = load_models(ROOT / "models.json")
        server = SokobanServer(("127.0.0.1", args.port), models)
    except (OSError, ValueError) as error:
        parser.exit(1, f"启动失败：{error}\n")
    with server:
        print(f"推箱子实验室 · http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
