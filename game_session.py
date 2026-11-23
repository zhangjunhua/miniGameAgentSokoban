"""A run keeps the board, execution policy, recent feedback and metrics."""

import base64
import binascii
import json
import struct
from collections import deque
from dataclasses import asdict
from threading import Lock
from uuid import uuid4

from model_client import Model, ModelClient, ModelError, redact_api_keys
from sokoban import DIRECTIONS, LEVELS, Level, first_static_deadlock, move, solve, won

ACTION_LIMIT = 40
CALL_LIMIT = 40
STRATEGIES = ("single", "plan_one", "plan_all")
DIRECTION_NAMES = {"up": "上", "down": "下", "left": "左", "right": "右"}
MOVE_CHOICES = {direction: f"向{name}移动一格" for direction, name in DIRECTION_NAMES.items()}
OUTCOME_NAMES = {"moved": "移动成功", "pushed": "推动箱子", "blocked": "未能移动"}
TEXT_LEGEND = "# 墙，空格 地板，@ 玩家，$ 箱子，. 目标，+ 玩家站在目标上，* 箱子在目标上。"
IMAGE_LEGEND = "蓝灰色砖墙是墙，深色格子是地板，浅蓝色圆形是玩家，棕色方箱是箱子，青色圆环是目标。青色方箱表示箱子已在目标上，玩家外的青色圆环表示玩家站在目标上。"
GAME_RULES = (
    "你在玩推箱子。每次沿上下左右移动一格。墙不能穿过。"
    "箱子只能推不能拉，一次只能推一个箱子，且箱子前方必须为空。"
    "让所有箱子到达目标即可通关。注意把箱子推入无法拉回的位置会造成死锁。"
)


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def validate_png(image) -> str:
    prefix = "data:image/png;base64,"
    if not isinstance(image, str) or not image.startswith(prefix) or len(image) > 1_000_000:
        raise RequestError("请提供当前棋盘的 PNG 截图。")
    try:
        png = base64.b64decode(image[len(prefix):], validate=True)
        width, height = struct.unpack(">II", png[16:24])
    except (binascii.Error, struct.error, ValueError):
        raise RequestError("棋盘截图格式无效。") from None
    if png[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" or not (0 < width <= 1024 and 0 < height <= 1024):
        raise RequestError("棋盘截图尺寸或格式无效。")
    return image


def parse_plan(text: str, horizon: int) -> tuple[list[str], str]:
    lines = text.strip().splitlines()
    fence = chr(96) * 3
    if len(lines) >= 3 and lines[0].lower() in (fence, fence + "json") and lines[-1] == fence:
        text = "\n".join(lines[1:-1])
    try:
        plan = json.loads(text)
    except (ValueError, RecursionError):
        plan = None
    actions = plan.get("actions") if isinstance(plan, dict) else None
    if (
        not isinstance(actions, list)
        or not 1 <= len(actions) <= horizon
        or any(not isinstance(action, str) or action not in DIRECTIONS for action in actions)
    ):
        raise ValueError(f"模型应返回包含 1–{horizon} 个有效方向的 JSON 动作计划。")
    note = plan.get("note", "")
    if not isinstance(note, str):
        raise ValueError("模型返回的动作说明应为文本。")
    return actions, note.strip()[:240]


class Run:
    def __init__(
        self, level: Level, mode: str, model: Model | None = None, observation="text",
        strategy="single", horizon=4, *, deadlock_guard: bool = False,
    ):
        self.id = uuid4().hex
        self.level = level
        self.mode = mode
        self.model = model
        self.observation = observation
        self.strategy = strategy
        self.horizon = 1 if strategy == "single" else horizon
        self.deadlock_guard = deadlock_guard
        self.last_rejection = ""
        self.board = level.initial
        self.status = "solved" if won(level, self.board) else "running"
        self.revision = 0
        self.lock = Lock()
        self.busy = False
        self.pending = deque()
        if mode == "bfs":
            path = solve(level)
            if path is None:
                raise RequestError("BFS 搜索不到通关路线。", 422)
            self.pending.extend(path)
        self.recent = deque(maxlen=4)
        self.moves = self.pushes = self.blocked = self.calls = self.wait_ms = 0
        self.input_tokens = self.output_tokens = self.usage_calls = 0
        self.rejected_plans = 0

    def snapshot(self) -> dict:
        level = self.level
        return {
            "id": self.id,
            "level": level.id,
            "mode": self.mode,
            "model": self.model.public() if self.model else None,
            "observation": self.observation,
            "strategy": self.strategy,
            "horizon": self.horizon,
            "deadlockGuard": self.deadlock_guard,
            "board": level.text(self.board),
            "status": self.status,
            "revision": self.revision,
            "pending": list(self.pending),
            "needsObservation": self.mode == "model" and self.status == "running" and not self.pending,
            "metrics": {
                "moves": self.moves,
                "pushes": self.pushes,
                "blocked": self.blocked,
                "calls": self.calls,
                "waitMs": self.wait_ms,
                "inputTokens": self.input_tokens,
                "outputTokens": self.output_tokens,
                "usageCalls": self.usage_calls,
                "rejectedPlans": self.rejected_plans,
            },
            "goals": len(level.goals),
            "filledGoals": len(self.board.boxes & level.goals),
        }

    def end(self) -> dict:
        with self.lock:
            if self.status == "running":
                self.status = "ended"
                self.pending.clear()
                self.revision += 1
            return self.snapshot()

    def step(self, payload: dict, client: ModelClient) -> dict:
        with self.lock:
            if self.status != "running":
                raise RequestError("这局已经结束。", 409)
            if self.busy or type(payload.get("revision")) is not int or payload["revision"] != self.revision:
                raise RequestError("棋盘已经更新，请以当前画面为准。", 409)
            if self.mode == "manual":
                direction = payload.get("direction")
                if not isinstance(direction, str) or direction not in DIRECTIONS:
                    raise RequestError("请选择上、下、左或右。")
                return self.advance(direction)
            if self.pending:
                return self.advance(self.pending.popleft())

            image = validate_png(payload.get("image")) if self.observation == "image" else None
            board = self.level.text(self.board)
            instructions, observation = self.prompt(board)
            self.calls += 1
            self.busy = True

        completion = None
        api_trace = None
        actions = []
        note = ""
        error = ""
        wait_ms = 0
        try:
            context = {
                "run_id": self.id,
                "level": self.level.id,
                "plan": self.calls,
                "observation": self.observation,
                "strategy": self.strategy,
                "horizon": self.horizon,
            }
            if self.deadlock_guard:
                context["deadlock_guard"] = True
            decision = self.model.protocol == "decisions"
            if decision:
                completion = client.choose(self.model, instructions, observation, image, MOVE_CHOICES, context=context)
            else:
                completion = client.complete(self.model, instructions, observation, image, context=context)
            api_trace = completion.trace
            wait_ms = completion.elapsed_ms
            actions, note = ([completion.text], "") if decision else parse_plan(completion.text, self.horizon)
            note = redact_api_keys(note)
        except ModelError as problem:
            error = str(problem)
            wait_ms = problem.elapsed_ms
            api_trace = problem.trace
        except ValueError as problem:
            error = str(problem)
        except Exception:
            with self.lock:
                self.busy = False
            raise

        with self.lock:
            self.busy = False
            if self.status != "running":
                raise RequestError("本局已结束，旧计划已作废。", 409)
            self.wait_ms += wait_ms
            plan = {
                "id": self.calls, "actions": actions, "note": note, "board": board,
                "model": completion.model if completion else self.model.model, "waitMs": wait_ms,
                "error": error,
            }
            if api_trace is not None:
                plan["api"] = asdict(api_trace)
            if completion and completion.input_tokens is not None and completion.output_tokens is not None:
                self.input_tokens += completion.input_tokens
                self.output_tokens += completion.output_tokens
                self.usage_calls += 1
                plan["tokens"] = [completion.input_tokens, completion.output_tokens]
            if error:
                if completion and api_trace is None:
                    plan["raw"] = redact_api_keys(completion.text)[:800]
            elif self.deadlock_guard:
                rejection = self.check_deadlock(actions)
                if rejection is not None:
                    plan["rejection"] = rejection
            if error or "rejection" in plan:
                self.update_status()
                self.revision += 1
                return {"run": self.snapshot(), "plan": plan, "action": None}
            if self.strategy == "plan_all":
                self.pending.extend(actions[1:])
            return self.advance(actions[0], plan)

    def check_deadlock(self, actions: list[str]) -> dict | None:
        remaining = ACTION_LIMIT - self.moves - self.blocked
        executable = actions[:remaining] if self.strategy == "plan_all" else actions[:1]
        deadlock = first_static_deadlock(self.level, self.board, executable)
        if deadlock is None:
            self.last_rejection = ""
            return None

        step, (row, column) = deadlock
        direction = actions[step - 1]
        self.rejected_plans += 1
        self.last_rejection = (
            f"第 {step} 步向{DIRECTION_NAMES[direction]}会把箱子推入第 {row + 1} 行、第 {column + 1} 列的静态死格，"
            "该格的箱子无法到达任何目标。整份计划未执行，请从当前棋盘改用其他路线。"
        )
        return {
            "step": step,
            "direction": direction,
            "destination": [row + 1, column + 1],
            "reason": self.last_rejection,
        }

    def prompt(self, board: list[str]) -> tuple[str, str]:
        task = (
            "请决定下一次移动的一步方向。" if self.strategy == "single" else
            f"请规划接下来的 {self.horizon} 步。如果预计更早通关，可以给出较短计划。"
        )
        if self.model is not None and self.model.protocol == "decisions":
            instructions = GAME_RULES + "根据当前棋盘和最近动作反馈，选择最有助于通关的下一步。"
        else:
            instructions = (
                f"{GAME_RULES}{task} 只返回 JSON 对象，格式为 {{\"actions\":[\"right\"],\"note\":\"可选的一句动作说明\"}}。"
                "方向只能用 up、down、left、right。"
            )
        feedback = "；".join(
            f"{DIRECTION_NAMES[item['direction']]} {item['reason'] or OUTCOME_NAMES[item['outcome']]}"
            for item in self.recent
        ) or "无"
        legend = TEXT_LEGEND if self.observation == "text" else IMAGE_LEGEND
        observation = f"棋盘图例：{legend}\n最近动作反馈：{feedback}\n"
        if self.last_rejection:
            observation += f"上次死锁预演：{self.last_rejection}\n"
        observation += "当前棋盘："
        if self.observation == "text":
            observation += "\n" + "\n".join(board)
        return instructions, observation

    def advance(self, direction: str, plan: dict | None = None) -> dict:
        result = move(self.level, self.board, direction)
        self.board = result.board
        if result.outcome == "blocked":
            self.blocked += 1
            self.pending.clear()
        else:
            self.moves += 1
            self.pushes += int(result.outcome == "pushed")
        feedback = {"direction": direction, "outcome": result.outcome, "reason": result.reason}
        self.recent.append(feedback)
        self.update_status()
        self.revision += 1
        action = {**feedback, "planId": self.calls if self.mode == "model" else None}
        return {"run": self.snapshot(), "plan": plan, "action": action}

    def update_status(self):
        if won(self.level, self.board):
            self.status = "solved"
        elif self.mode == "model" and (
            self.moves + self.blocked >= ACTION_LIMIT or (self.calls >= CALL_LIMIT and not self.pending)
        ):
            self.status = "limit"
        if self.status != "running":
            self.pending.clear()


class RunStore:
    def __init__(self, models: dict[str, Model]):
        self.models = models
        self.runs: dict[str, Run] = {}
        self.lock = Lock()

    def get(self, run_id: str) -> Run:
        with self.lock:
            run = self.runs.get(run_id)
        if run is None:
            raise RequestError("本局已不存在，请开始新局。", 404)
        return run

    def create(self, payload: dict) -> dict:
        level_id = payload.get("level")
        mode = payload.get("mode")
        if not isinstance(level_id, str) or level_id not in LEVELS or mode not in ("manual", "bfs", "model"):
            raise RequestError("请选择有效的关卡与玩法。")
        model = None
        observation, strategy, horizon = "text", "single", 4
        deadlock_guard = False
        if mode == "model":
            model_id = payload.get("model")
            model = self.models.get(model_id) if isinstance(model_id, str) else None
            observation = payload.get("observation")
            strategy = payload.get("strategy")
            horizon = payload.get("horizon", 4)
            deadlock_guard = payload.get("deadlockGuard", False)
            if model is None or not model.available:
                raise RequestError("请选择已配置密钥的模型。")
            if observation not in ("text", "image") or (observation == "image" and not model.image):
                raise RequestError("此模型不支持所选棋盘输入。")
            if strategy not in STRATEGIES or type(horizon) is not int or not 2 <= horizon <= 8:
                raise RequestError("请选择有效的走棋策略和规划长度。")
            if model.protocol == "decisions" and strategy != "single":
                raise RequestError("决策模型每次选择一个方向，请使用单步决策。")
            if type(deadlock_guard) is not bool:
                raise RequestError("死锁预演设置应为开启或关闭。")
        previous_id = payload.get("previous")
        if previous_id is not None and not isinstance(previous_id, str):
            raise RequestError("旧局标识无效。")
        run = Run(LEVELS[level_id], mode, model, observation, strategy, horizon, deadlock_guard=deadlock_guard)
        with self.lock:
            previous = self.runs.pop(previous_id, None)
            previous_state = previous.end() if previous else None
            self.runs[run.id] = run
        return {"run": run.snapshot(), "previous": previous_state}
