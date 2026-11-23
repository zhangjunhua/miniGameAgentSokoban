import json
import os
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app import SokobanServer
from game_session import RequestError, Run, RunStore
from model_client import DECISIONS_ENDPOINT, Completion, Model, ModelClient, ModelError, load_local_keys, load_models
from model_log import ModelCallLog
from sokoban import LEVELS, first_static_deadlock, move, solve, won

MODEL = Model("openai", "test-vision", "Test Vision", True)
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jEgAAAABJRU5ErkJggg=="


class ScriptedModel:
    def __init__(self, *replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, model, instructions, observation, image, *, context=None):
        self.requests.append((model, instructions, observation, image))
        actions = next(self.replies)
        if isinstance(actions, Exception):
            raise actions
        text = json.dumps({"actions": actions}) if isinstance(actions, list) else actions
        return Completion(text, model.model, 12, 60, 10)


def step(run, api, **payload):
    return run.step({"revision": run.revision, **payload}, api)


class RuleTests(unittest.TestCase):
    def test_static_deadlock_check_keeps_all_five_reference_routes(self):
        for level in LEVELS.values():
            with self.subTest(level=level.id):
                self.assertIsNone(first_static_deadlock(level, level.initial, list(solve(level))))
        level = LEVELS["3"]
        self.assertEqual(first_static_deadlock(level, level.initial, ["up"]), (1, (1, 3)))

    def test_all_reference_routes_reach_the_goal_in_the_expected_minimum(self):
        for level, distance in zip(LEVELS.values(), (1, 6, 8, 11, 24), strict=True):
            with self.subTest(level=level.id):
                board = level.initial
                path = solve(level)
                self.assertEqual(len(path), distance)
                for direction in path:
                    result = move(level, board, direction)
                    self.assertNotEqual(result.outcome, "blocked")
                    board = result.board
                self.assertTrue(won(level, board))

    def test_wall_and_two_box_blocking(self):
        level = LEVELS["4"]
        board = move(level, move(level, level.initial, "left").board, "up").board
        result = move(level, board, "right")
        self.assertEqual(result.outcome, "blocked")
        self.assertEqual(result.board, board)
        level = LEVELS["1"]
        at_wall = move(level, level.initial, "up").board
        self.assertEqual(move(level, at_wall, "up").board, at_wall)

    def test_level_three_allows_a_push_into_a_dead_square(self):
        level = LEVELS["3"]
        result = move(level, level.initial, "up")
        self.assertEqual(result.outcome, "pushed")
        self.assertIsNone(solve(level, result.board))


class StrategyTests(unittest.TestCase):
    def test_guard_rejects_the_whole_plan_then_replans_with_feedback(self):
        level = LEVELS["2"]
        route = list(solve(level))
        client = ScriptedModel(["right", "right"], route)
        run = Run(level, "model", MODEL, strategy="plan_all", horizon=8, deadlock_guard=True)

        rejected = step(run, client)
        self.assertIsNone(rejected["action"])
        self.assertEqual(rejected["plan"]["error"], "")
        self.assertEqual(rejected["plan"]["actions"], ["right", "right"])
        self.assertEqual(rejected["plan"]["rejection"]["step"], 2)
        self.assertEqual(rejected["plan"]["rejection"]["destination"], [3, 7])
        self.assertEqual(rejected["run"]["board"], level.text(level.initial))
        self.assertEqual(rejected["run"]["pending"], [])
        self.assertEqual(rejected["run"]["revision"], 1)
        self.assertEqual(rejected["run"]["metrics"]["rejectedPlans"], 1)
        self.assertEqual(rejected["run"]["metrics"]["moves"], 0)
        self.assertEqual(rejected["run"]["metrics"]["blocked"], 0)
        self.assertEqual(rejected["run"]["metrics"]["calls"], 1)

        recovered = step(run, client)
        self.assertEqual(recovered["action"]["direction"], route[0])
        self.assertIn("上次死锁预演", client.requests[1][2])
        self.assertIn("第 3 行、第 7 列", client.requests[1][2])
        self.assertIn("\n".join(level.text(level.initial)), client.requests[1][2])
        for _ in route[1:]:
            recovered = step(run, client)
        self.assertEqual(recovered["run"]["status"], "solved")
        self.assertEqual(recovered["run"]["metrics"]["calls"], 2)

    def test_preview_respects_execution_policy_and_stops_at_obstacles(self):
        level = LEVELS["2"]
        run = Run(level, "model", MODEL, strategy="plan_one", deadlock_guard=True)
        chosen = step(run, ScriptedModel(["right", "right"]))
        self.assertEqual(chosen["action"]["outcome"], "pushed")
        self.assertNotIn("rejection", chosen["plan"])

        run = Run(level, "model", MODEL, strategy="plan_all", deadlock_guard=True)
        blocked = ScriptedModel(["up", "up", "right", "right"])
        first = step(run, blocked)
        self.assertNotIn("rejection", first["plan"])
        second = step(run, blocked)
        self.assertEqual(second["action"]["outcome"], "blocked")
        self.assertEqual(second["run"]["pending"], [])

    def test_default_leaves_a_deadlocking_move_available(self):
        level = LEVELS["3"]
        result = step(Run(level, "model", MODEL), ScriptedModel(["up"]))
        self.assertFalse(result["run"]["deadlockGuard"])
        self.assertEqual(result["action"]["outcome"], "pushed")
        self.assertNotIn("rejection", result["plan"])

    def test_single_decision_reads_a_fresh_board_and_feedback(self):
        client = ScriptedModel(["up"], ["up"])
        run = Run(LEVELS["1"], "model", MODEL)
        start_board = run.snapshot()["board"]
        first = step(run, client)
        second = step(run, client)
        self.assertEqual(first["plan"]["board"], start_board)
        self.assertEqual(first["action"]["outcome"], "moved")
        self.assertEqual(second["action"]["outcome"], "blocked")
        self.assertIn("只能", client.requests[0][1])
        self.assertIn("\n".join(first["run"]["board"]), client.requests[1][2])
        self.assertIn("上 移动成功", client.requests[1][2])
        self.assertEqual(second["run"]["metrics"]["calls"], 2)

    def test_planning_one_step_looks_ahead_but_refreshes_the_plan(self):
        client = ScriptedModel(["up", "up", "right", "down"], ["down", "right"])
        run = Run(LEVELS["1"], "model", MODEL, strategy="plan_one")
        first = step(run, client)
        second = step(run, client)
        self.assertEqual(first["plan"]["actions"], ["up", "up", "right", "down"])
        self.assertEqual(first["run"]["pending"], [])
        self.assertEqual(second["action"]["direction"], "down")
        self.assertEqual(len(client.requests), 2)

    def test_batch_uses_queued_move_and_replans_on_the_first_block(self):
        client = ScriptedModel(["up", "up", "right", "down"], ["down", "right", "left"])
        run = Run(LEVELS["1"], "model", MODEL, strategy="plan_all")
        first = step(run, client)
        blocked = step(run, client)
        self.assertEqual(first["run"]["pending"], ["up", "right", "down"])
        self.assertEqual(blocked["action"]["outcome"], "blocked")
        self.assertEqual(blocked["action"]["planId"], first["plan"]["id"])
        self.assertIsNone(blocked["plan"])
        self.assertEqual(blocked["run"]["pending"], [])
        self.assertEqual(len(client.requests), 1)
        step(run, client)
        self.assertIn("前方是墙", client.requests[-1][2])
        solved = step(run, client)
        self.assertEqual(solved["run"]["status"], "solved")
        self.assertEqual(solved["run"]["pending"], [])
        self.assertEqual(solved["run"]["metrics"]["calls"], 2)

    def test_image_is_required_only_for_a_new_image_plan(self):
        client = ScriptedModel(["up", "down"])
        run = Run(LEVELS["1"], "model", MODEL, observation="image", strategy="plan_all")
        with self.assertRaises(RequestError):
            step(run, client, image="invalid")
        first = step(run, client, image=PNG)
        self.assertEqual(client.requests[0][3], PNG)
        self.assertNotIn(first["plan"]["board"][0], client.requests[0][2])
        step(run, client)
        self.assertEqual(len(client.requests), 1)

    def test_bad_plan_and_timeout_keep_the_board_and_count_calls(self):
        client = ScriptedModel(["up", "right"], ModelError("请求超时", 30000), ["right"])
        run = Run(LEVELS["1"], "model", MODEL)
        initial = run.board
        for _ in range(2):
            result = step(run, client)
            self.assertIsNone(result["action"])
            self.assertTrue(result["plan"]["error"])
            self.assertEqual(run.board, initial)
        result = step(run, client)["run"]
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["metrics"]["calls"], 3)
        self.assertEqual(result["metrics"]["waitMs"], 30024)

    def test_action_budget_counts_blocked_moves_and_ends_without_extra_calls(self):
        client = ScriptedModel(*([["up"]] * 40))
        run = Run(LEVELS["1"], "model", MODEL)
        for _ in range(40):
            result = step(run, client)
        self.assertEqual(result["run"]["status"], "limit")
        self.assertEqual(result["run"]["metrics"]["blocked"], 39)
        with self.assertRaises(RequestError):
            step(run, client)
        self.assertEqual(len(client.requests), 40)

    def test_bfs_replay_does_not_call_a_model(self):
        run = Run(LEVELS["2"], "bfs")
        for _ in range(6):
            result = step(run, None)["run"]
        self.assertEqual(result["status"], "solved")
        self.assertEqual(result["metrics"]["moves"], 6)


class SessionTests(unittest.TestCase):
    def test_replacement_invalidates_an_in_flight_plan(self):
        entered = Event()
        release = Event()
        failures = []

        class SlowModel:
            def complete(self, *args, **kwargs):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("Test request was never released")
                return Completion('{"actions":["right"]}', MODEL.model, 10, 3, 3)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            store = RunStore({MODEL.id: MODEL})
            config = {"level": "1", "mode": "model", "model": MODEL.id, "observation": "text", "strategy": "single"}
            first = store.create(config)["run"]
            run = store.get(first["id"])

            def request_step():
                try:
                    step(run, SlowModel())
                except Exception as error:
                    failures.append(error)

            worker = Thread(target=request_step, daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaises(RequestError):
                    step(run, SlowModel())
                replacement = store.create({**config, "previous": run.id})
                self.assertEqual(replacement["previous"]["status"], "ended")
                self.assertEqual(replacement["run"]["board"], first["board"])
            finally:
                release.set()
                worker.join(3)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RequestError)
        self.assertEqual(run.board, run.level.initial)

    def test_duplicate_revision_cannot_move_twice(self):
        run = Run(LEVELS["1"], "manual")
        step(run, None, direction="up")
        with self.assertRaises(RequestError):
            run.step({"revision": 0, "direction": "down"}, None)
        self.assertEqual(run.moves, 1)


class ModelClientTests(unittest.TestCase):
    def test_jev_selects_one_move_with_text_or_image_and_preserves_the_api_trace(self):
        models = load_models(Path(__file__).with_name("models.json"))
        jev = models["openrouter:typesafe/jev-1.14"]
        raw = json.dumps({
            "model": "typesafe/jev-1.14-20261014",
            "answers": {"move": {"type": "choice", "choice": "right", "probabilities": {"right": 0.9}}},
            "usage": {"input_tokens": 256, "output_tokens": 23},
        })
        requests = []

        def receive(request, timeout):
            requests.append(request)
            response = BytesIO(raw.encode())
            response.status = 200
            return response

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "jev-fixture-key"}):
            with TemporaryDirectory() as directory:
                call_log = ModelCallLog(Path(directory))
                client = ModelClient(receive, call_log=call_log)
                for observation, image in (("text", None), ("image", PNG)):
                    with self.subTest(observation=observation):
                        run = Run(LEVELS["1"], "model", jev, observation=observation)
                        result = step(run, client, image=image)
                        request = requests[-1]
                        payload = json.loads(request.data)
                        self.assertEqual(request.full_url, DECISIONS_ENDPOINT)
                        self.assertEqual(request.get_header("Authorization"), "Bearer jev-fixture-key")
                        self.assertEqual(set(payload), {"model", "state", "questions"})
                        self.assertEqual(set(payload["questions"]["move"]["criteria"]), {"up", "down", "left", "right"})
                        self.assertNotIn("JSON", payload["questions"]["move"]["instructions"])
                        if image:
                            self.assertEqual(payload["state"][1]["image_url"]["url"], PNG)
                        else:
                            self.assertIn("# @$. #", payload["state"])
                        self.assertEqual(result["plan"]["actions"], ["right"])
                        self.assertEqual(result["plan"]["tokens"], [256, 23])
                        self.assertEqual(result["plan"]["api"]["response"], raw)
                        self.assertEqual(result["run"]["status"], "solved")
                records = [json.loads(line) for line in call_log.path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 4)
        self.assertEqual(records[-2]["call_id"], records[-1]["call_id"])
        self.assertEqual(records[-1]["call_id"], result["plan"]["api"]["call_id"])

    def test_decisions_reject_invalid_choices_and_multiple_steps(self):
        jev = Model("openrouter", "typesafe/jev-1.14", "Jev", True, "decisions")
        response = b'{"answers":{"move":{"type":"choice","choice":"jump"}}}'
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "jev-fixture-key"}):
            run = Run(LEVELS["1"], "model", jev)
            result = step(run, ModelClient(lambda request, timeout: BytesIO(response)))
            with self.assertRaisesRegex(RequestError, "单步决策"):
                RunStore({jev.id: jev}).create({
                    "level": "1", "mode": "model", "model": jev.id, "observation": "text", "strategy": "plan_all",
                })

        self.assertIsNone(result["action"])
        self.assertEqual(result["run"]["board"], LEVELS["1"].text(LEVELS["1"].initial))
        self.assertIn("没有返回可选动作", result["plan"]["error"])
        self.assertEqual(result["plan"]["api"]["response"], response.decode())

    def test_local_keys_fill_missing_provider_credentials_only(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "temporary"}, clear=True):
            config = Path(directory) / ".env"
            config.write_text(
                "# Local provider keys\nOPENAI_API_KEY=file-value\nOPENROUTER_API_KEY = router-value\nOTHER_KEY=ignored\n",
                encoding="utf-8-sig",
            )
            load_local_keys(config)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "temporary")
            self.assertEqual(os.environ["OPENROUTER_API_KEY"], "router-value")
            self.assertNotIn("OTHER_KEY", os.environ)

    def test_missing_local_file_leaves_credentials_unchanged(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            load_local_keys(Path(directory) / ".env")
            self.assertFalse(MODEL.available)

    def test_adapters_share_payload_shape_and_keep_credentials_separate(self):
        requests = []

        def receive(request, timeout):
            requests.append((request, timeout))
            return BytesIO(b'{"model":"served-model","choices":[{"message":{"content":"{}"}}],"usage":{"prompt_tokens":3,"completion_tokens":4}}')

        router = Model("openrouter", "author/example", "Example", True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key-one", "OPENROUTER_API_KEY": "key-two"}):
            client = ModelClient(receive)
            for model, key, host in (
                (MODEL, "key-one", "api.openai.com"),
                (router, "key-two", "openrouter.ai"),
            ):
                for image in (None, PNG):
                    answer = client.complete(model, "instructions", "observation", image)
                    self.assertEqual((answer.model, answer.input_tokens, answer.output_tokens), ("served-model", 3, 4))
                    request, timeout = requests[-1]
                    self.assertEqual(request.host, host)
                    self.assertEqual(request.get_header("Authorization"), "Bearer " + key)
                    self.assertEqual(timeout, 30)
                    payload = json.loads(request.data)
                    self.assertEqual(payload["model"], model.model)
                    user = payload["messages"][1]["content"]
                    self.assertEqual(user[1]["image_url"]["url"] if image else user, image or "observation")

    def test_plan_preserves_api_bodies_for_the_screenshot_that_was_sent(self):
        requests = []
        raw = json.dumps({
            "choices": [{"message": {"content": '{"actions":["right"],"note":"Next move"}'}}],
            "usage": {"prompt_tokens": 75, "completion_tokens": 10},
        }).encode()

        def receive(request, timeout):
            requests.append(request)
            response = BytesIO(raw)
            response.status = 200
            return response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "trace-fixture-key"}):
            run = Run(LEVELS["1"], "model", MODEL, observation="image")
            result = step(run, ModelClient(receive), image=PNG)

        trace = result["plan"]["api"]
        self.assertEqual(result["run"]["status"], "solved")
        self.assertEqual(trace["status"], 200)
        self.assertEqual(trace["endpoint"], requests[0].full_url)
        self.assertEqual(trace["request"].encode(), requests[0].data)
        self.assertEqual(json.loads(trace["request"])["messages"][1]["content"][1]["image_url"]["url"], PNG)
        self.assertEqual(trace["response"], raw.decode())
        self.assertNotIn("trace-fixture-key", json.dumps(result))

    def test_http_error_shows_diagnostics_but_redacts_an_echoed_key(self):
        key = "trace-fixture-key"
        body = BytesIO(json.dumps({"error": {"message": f"Invalid credential {key}"}}).encode())

        def reject(request, timeout):
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, body)

        with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
            result = step(Run(LEVELS["1"], "model", MODEL), ModelClient(reject))

        trace = result["plan"]["api"]
        self.assertIsNone(result["action"])
        self.assertEqual(trace["status"], 401)
        self.assertIn("Invalid credential [密钥已隐藏]", trace["response"])
        self.assertNotIn(key, json.dumps(result))

    def test_malformed_provider_response_remains_inspectable(self):
        def receive(request, timeout):
            response = BytesIO(b"not valid JSON")
            response.status = 200
            return response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "trace-fixture-key"}):
            result = step(Run(LEVELS["1"], "model", MODEL), ModelClient(receive))

        self.assertIsNone(result["action"])
        self.assertIn("无法读取", result["plan"]["error"])
        self.assertEqual(result["plan"]["api"]["response"], "not valid JSON")

    def test_transport_reports_timeout_without_exposing_exception_details(self):
        def timeout(*args, **kwargs):
            raise URLError(TimeoutError("internal detail"))

        with TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test-credential"}):
            call_log = ModelCallLog(Path(directory) / "log")
            with self.assertRaisesRegex(ModelError, "请求超时") as error:
                ModelClient(timeout, call_log=call_log).complete(MODEL, "rules", "data", None)
            raw = call_log.path.read_text(encoding="utf-8")
            started, ended = map(json.loads, raw.splitlines())
        trace = error.exception.trace
        self.assertIsNone(trace.status)
        self.assertIsNone(trace.response)
        self.assertEqual(json.loads(trace.request)["messages"][1]["content"], "data")
        self.assertEqual(started["call_id"], ended["call_id"])
        self.assertEqual(ended["event"], "response")
        self.assertIsNone(ended["body"])
        self.assertIn("超时", ended["error"])
        self.assertNotIn("internal detail", raw)

    def test_request_is_not_sent_when_its_log_cannot_be_written(self):
        def receive(*args, **kwargs):
            self.fail("The provider should only be called after the request is written")

        with TemporaryDirectory() as directory, patch.dict(os.environ, {"OPENAI_API_KEY": "test-credential"}):
            blocked = Path(directory) / "log"
            blocked.write_text("not a directory", encoding="utf-8")
            client = ModelClient(receive, call_log=ModelCallLog(blocked))
            with self.assertRaisesRegex(ModelError, "请求未发送"):
                client.complete(MODEL, "rules", "data", None)


class HttpTests(unittest.TestCase):
    def test_model_endpoint_persists_both_events_under_the_ui_call_id(self):
        key = "trace-fixture-key"

        with TemporaryDirectory() as directory, patch("app.ROOT", Path(directory)), patch.dict(
            os.environ, {"OPENAI_API_KEY": key}
        ):
            with SokobanServer(("127.0.0.1", 0), {MODEL.id: MODEL}) as server:
                log_path = server.client.call_log.path
                received = []
                initial_records = []

                def provider(request, timeout):
                    initial_records.extend(map(json.loads, log_path.read_text(encoding="utf-8").splitlines()))
                    received.append(request.data.decode())
                    reply = {"actions": ["right"], "note": key}
                    response = BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(reply)}}]}).encode())
                    response.status = 200
                    return response

                server.client.open_request = provider
                thread = Thread(target=server.serve_forever, daemon=True)
                thread.start()
                root = f"http://127.0.0.1:{server.server_port}"

                def post(path, body):
                    request = Request(root + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
                    with urlopen(request, timeout=4) as response:
                        return json.load(response)

                try:
                    run = post("/api/runs", {
                        "level": "1", "mode": "model", "model": MODEL.id,
                        "observation": "image", "strategy": "single",
                    })["run"]
                    result = post(f"/api/runs/{run['id']}/step", {"revision": 0, "image": PNG})
                    raw = log_path.read_text(encoding="utf-8")
                finally:
                    server.shutdown()
                    thread.join(4)

        started, ended = map(json.loads, raw.splitlines())
        self.assertEqual(initial_records, [started])
        self.assertEqual((started["event"], ended["event"]), ("request", "response"))
        self.assertEqual(started["body"], received[0])
        self.assertEqual(started["context"], {
            "run_id": run["id"], "level": "1", "plan": 1,
            "observation": "image", "strategy": "single", "horizon": 1,
        })
        self.assertEqual(ended["status"], 200)
        self.assertEqual(started["call_id"], ended["call_id"])
        self.assertEqual(ended["call_id"], result["plan"]["api"]["call_id"])
        self.assertEqual(ended["body"], result["plan"]["api"]["response"])
        self.assertEqual(result["run"]["status"], "solved")
        self.assertNotIn(key, raw)
        self.assertNotIn("Authorization", raw)

    def test_local_game_endpoints(self):
        with SokobanServer(("127.0.0.1", 0), {}) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            root = f"http://127.0.0.1:{server.server_port}"

            def call(path, obj=None):
                body = json.dumps(obj).encode() if obj is not None else None
                with urlopen(Request(root + path, body, {"Content-Type": "application/json"}), timeout=4) as res:
                    return json.load(res)

            try:
                with urlopen(root, timeout=4) as response:
                    self.assertIn("推箱子实验室", response.read().decode("utf-8"))
                self.assertEqual(len(call("/api/levels")["levels"]), 5)
                run = call("/api/runs", {"level": "1", "mode": "manual"})["run"]
                endpoint = f"/api/runs/{run['id']}/step"
                result = call(endpoint, {"revision": 0, "direction": "right"})
                self.assertEqual(result["run"]["status"], "solved")
            finally:
                server.shutdown()
                thread.join(4)


if __name__ == "__main__":
    unittest.main()
