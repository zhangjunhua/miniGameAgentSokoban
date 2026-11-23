"""Immutable Sokoban rules and a shortest-move reference solver."""

from collections import deque
from dataclasses import dataclass
from functools import cache

DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
Point = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Board:
    player: Point
    boxes: frozenset[Point]


@dataclass(frozen=True, slots=True)
class Level:
    id: str
    name: str
    description: str
    height: int
    width: int
    floor: frozenset[Point]
    goals: frozenset[Point]
    initial: Board

    def text(self, board: Board) -> list[str]:
        rows = []
        for row in range(self.height):
            tiles = []
            for column in range(self.width):
                cell = row, column
                goal = cell in self.goals
                if cell == board.player:
                    tile = "+" if goal else "@"
                elif cell in board.boxes:
                    tile = "*" if goal else "$"
                elif goal:
                    tile = "."
                else:
                    tile = " " if cell in self.floor else "#"
                tiles.append(tile)
            rows.append("".join(tiles))
        return rows


@dataclass(frozen=True, slots=True)
class Move:
    board: Board
    outcome: str
    reason: str = ""


def parse_level(level_id: str, name: str, description: str, layout: str) -> Level:
    rows = layout.split("/")
    if len({len(row) for row in rows}) != 1 or set("".join(rows)) - set("# @$.*+"):
        raise ValueError(f"Invalid layout: {level_id}")
    cells = {(r, c): tile for r, row in enumerate(rows) for c, tile in enumerate(row)}
    players = [point for point, tile in cells.items() if tile in "@+"]
    floor = frozenset(point for point, tile in cells.items() if tile != "#")
    boxes = frozenset(point for point, tile in cells.items() if tile in "$*")
    goals = frozenset(point for point, tile in cells.items() if tile in ".*+")
    if len(players) != 1 or not boxes or len(boxes) != len(goals):
        raise ValueError(f"Invalid player, box, or goal count: {level_id}")
    return Level(level_id, name, description, len(rows), len(rows[0]), floor, goals, Board(players[0], boxes))


LEVELS = {
    level.id: level
    for level in (
        parse_level("1", "第一推", "把箱子推到目标上。", "#######/#     #/# @$. #/#     #/#######"),
        parse_level("2", "绕到箱后", "先找到能朝目标推箱的位置。", "########/#      #/# .@$  #/#   #  #/#      #/########"),
        parse_level("3", "死锁边缘", "推箱后，想一想还能从哪一侧接近它。", "########/#      #/#  $   #/#  @#  #/#    . #/########"),
        parse_level("4", "两箱互挡", "相邻的箱子会挡住彼此，试着调整顺序。", "########/#      #/# .$$. #/#  @   #/#      #/########"),
        parse_level("5", "三箱交错", "绕过隔墙，为三个箱子安排路线。", "#########/#       #/# .$# . #/#  $    #/#  @# $ #/#    .  #/#########"),
    )
}


def move(level: Level, board: Board, direction: str) -> Move:
    dr, dc = DIRECTIONS[direction]
    row, column = board.player
    target = row + dr, column + dc
    if target not in level.floor:
        return Move(board, "blocked", "前方是墙")
    if target not in board.boxes:
        return Move(Board(target, board.boxes), "moved")
    destination = row + 2 * dr, column + 2 * dc
    if destination not in level.floor or destination in board.boxes:
        return Move(board, "blocked", "箱子前方被挡住")
    return Move(Board(target, (board.boxes - {target}) | {destination}), "pushed")


def won(level: Level, board: Board) -> bool:
    return board.boxes == level.goals


@cache
def reachable_box_squares(level: Level) -> frozenset[Point]:
    """Overestimate where a single box can reach a goal, ignoring other boxes."""
    reachable = set(level.goals)
    queue = deque(level.goals)
    while queue:
        row, column = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            previous = row - dr, column - dc
            standing = row - 2 * dr, column - 2 * dc
            if previous in level.floor and standing in level.floor and previous not in reachable:
                reachable.add(previous)
                queue.append(previous)
    return frozenset(reachable)


def first_static_deadlock(level: Level, board: Board, actions: list[str]) -> tuple[int, Point] | None:
    """Find the first push into a static dead square before any block or win."""
    reachable = reachable_box_squares(level)
    for step, direction in enumerate(actions, 1):
        result = move(level, board, direction)
        if result.outcome == "blocked":
            break
        if result.outcome == "pushed":
            destination, = result.board.boxes - board.boxes
            if destination not in reachable:
                return step, destination
        board = result.board
        if won(level, board):
            break
    return None


@cache
def solve(level: Level, start: Board | None = None) -> tuple[str, ...] | None:
    """Return the shortest move sequence, or None if the queue is exhausted."""
    initial = level.initial if start is None else start
    parents = {initial: None}
    queue = deque([initial])
    while queue:
        board = queue.popleft()
        if won(level, board):
            path = []
            while parents[board] is not None:
                board, direction = parents[board]
                path.append(direction)
            return tuple(reversed(path))
        for direction in DIRECTIONS:
            next_board = move(level, board, direction).board
            if next_board not in parents:
                parents[next_board] = (board, direction)
                queue.append(next_board)
    return None
