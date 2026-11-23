"use strict";

const byId = (id) => document.getElementById(id);
const canvas = byId("board");
const ctx = canvas.getContext("2d");
const modes = {manual: "手动", bfs: "BFS 求解", model: "大模型"};
const strategies = {single: "单步决策", plan_one: "多步规划，走一步", plan_all: "多步规划，走整段"};
const statusLabels = {running: "进行中", solved: "通关", limit: "达到上限", ended: "已结束"};
const directions = {up: "上", down: "下", left: "左", right: "右"};
const keys = {ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right"};
const outcomes = {moved: "移动", pushed: "推箱", blocked: "被阻挡"};
const modelOptions = byId("model");
const observationOptions = byId("observation");
const strategyOptions = byId("strategy");
const horizonOptions = byId("horizon");
const deadlockGuard = byId("deadlock-guard");
const formatter = new Intl.NumberFormat("zh-CN");

let levels = [];
let models = [];
let run = null;
let frames = [];
let review = null;
let selectedMode = "manual";
let opening = false;
let flight = null;
let auto = false;
let timer = null;
let problem = "";
const results = new Map();

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "本地服务返回了错误。");
  return data;
}

function paint(color, x, y, w, h) {
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
}

function rounded(color, x, y, w, h, radius, stroke = false) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, radius);
  if (stroke) {
    ctx.strokeStyle = color;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function circle(color, x, y, radius, stroke = false) {
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  if (stroke) {
    ctx.strokeStyle = color;
    ctx.stroke();
  } else {
    ctx.fillStyle = color;
    ctx.fill();
  }
}

function drawBoard(rows) {
  if (!rows) return;
  const size = 70;
  canvas.width = rows[0].length * size;
  canvas.height = rows.length * size;
  rows.forEach((row, r) => {
    [...row].forEach((tile, c) => {
      const x = c * size;
      const y = r * size;
      const centerX = x + 35;
      const centerY = y + 35;
      if (tile === "#") {
        paint((r + c) % 2 ? "#35475f" : "#394c65", x, y, size, size);
        paint("#506681", x + 2, y + 2, 66, 2);
        paint("#29394d", x, y + 66, 70, 4);
        paint("#2f4157", x + 2, y + 34, 66, 3);
        paint("#4c607b", x + (r % 2 ? 18 : 49), y + 4, 2, 30);
        paint("#4c607b", x + (r % 2 ? 49 : 18), y + 37, 2, 29);
        paint("#2b3a51", x + 68, y, 2, 70);
        return;
      }
      paint((r + c) % 2 ? "#172130" : "#192434", x, y, size, size);
      paint("#253448", x, y, size, 1);
      paint("#253448", x, y, 1, size);

      if (".+*".includes(tile)) {
        circle("#20433f", centerX, centerY, 23);
        ctx.lineWidth = 3;
        circle("#78d8c4", centerX, centerY, 19, true);
        circle("#60bda9", centerX, centerY, 4);
      }
      if ("$*".includes(tile)) {
        const placed = tile === "*";
        rounded("#0b111b", x + 12, y + 15, 47, 47, 7);
        rounded(placed ? "#7ce3ca" : "#d79658", x + 11, y + 10, 48, 48, 7);
        ctx.lineWidth = 2;
        rounded(placed ? "#348677" : "#754424", x + 17, y + 16, 36, 36, 4, true);
        ctx.beginPath();
        ctx.moveTo(x + 22, y + 21);
        ctx.lineTo(x + 48, y + 47);
        ctx.moveTo(x + 48, y + 21);
        ctx.lineTo(x + 22, y + 47);
        ctx.stroke();
      }
      if ("@+".includes(tile)) {
        circle("#0d1724", centerX, centerY + 4, 20);
        circle("#abdcfa", centerX, centerY, 20);
        ctx.lineWidth = 3;
        if (tile === "+") circle("#73e5cb", centerX, centerY, 24, true);
        circle("#e5f6ff", centerX - 5, centerY - 8, 4);
        circle("#27435a", centerX - 6, centerY + 1, 2);
        circle("#27435a", centerX + 6, centerY + 1, 2);
      }
    });
  });
  canvas.setAttribute("aria-label", "推箱子棋盘，" + rows[0].length + " 列，" + rows.length + " 行");
}

function stop() {
  auto = false;
  clearTimeout(timer);
  timer = null;
}

function playable() {
  return run?.status === "running" && run.mode === selectedMode && !review && !opening;
}

function activeSettings() {
  if (!run) return {mode: "manual"};
  if (run.mode !== "model") return {mode: run.mode};
  return {
    mode: "model", model: run.model.id, observation: run.observation,
    strategy: run.strategy, horizon: run.strategy === "single" ? 4 : run.horizon,
    deadlockGuard: run.deadlockGuard,
  };
}

function remember(finalRun, history) {
  if (!finalRun || finalRun.status === "running") return;
  if (!history.some((frame) => frame.action || frame.plan) && finalRun.metrics.calls === 0) return;
  results.set(finalRun.id, {run: finalRun, frames: history.slice()});
  renderResults();
}

async function start(level, settings) {
  if (opening) return;
  stop();
  opening = true;
  problem = "";
  render();
  try {
    const data = await api("/api/runs", {level, ...settings, previous: run?.id});
    remember(data.previous, frames);
    flight = null;
    run = data.run;
    frames = [{run, action: null, plan: null}];
    review = null;
    selectedMode = run.mode;
  } catch (error) {
    problem = error.message;
  } finally {
    opening = false;
    render();
  }
}

async function endRun() {
  if (!playable()) return;
  stop();
  opening = true;
  render();
  try {
    const data = await api("/api/runs/" + run.id + "/end", {});
    flight = null;
    run = data.run;
    problem = "";
    remember(run, frames);
  } catch (error) {
    problem = error.message;
  } finally {
    opening = false;
    render();
  }
}

async function step(direction) {
  if (!playable() || flight) return;
  const request = {revision: run.revision, direction};
  if (run.needsObservation && run.observation === "image") {
    request.image = canvas.toDataURL("image/png");
  }
  const turn = {id: run.id, awaitingModel: run.needsObservation};
  problem = "";
  flight = turn;
  renderControls();
  renderFeedback(run);

  try {
    const data = await api("/api/runs/" + turn.id + "/step", request);
    if (flight !== turn || run.id !== turn.id) return;
    run = data.run;
    if (data.plan && request.image) data.plan.image = request.image;
    frames.push(data);
    if (data.plan?.error) {
      problem = data.plan.error;
      stop();
    }
    if (run.status !== "running") {
      stop();
      remember(run, frames);
    }
  } catch (error) {
    if (flight === turn) {
      problem = error.message;
      stop();
    }
  } finally {
    if (flight === turn) {
      flight = null;
      render();
      if (auto && playable()) timer = setTimeout(() => step(), 270);
    }
  }
}

function setContinuous() {
  if (auto) {
    stop();
    renderControls();
    return;
  }
  if (!playable() || flight || run.mode === "manual") return;
  auto = true;
  step();
}

function changeMode(mode) {
  stop();
  selectedMode = mode;
  problem = "";
  if (run && mode !== "model" && run.mode !== mode) start(run.level, {mode});
  else render();
}

function renderModels() {
  const previous = modelOptions.value;
  const useImage = observationOptions.value === "image";
  modelOptions.replaceChildren();
  for (const model of models.filter((entry) => !useImage || entry.image)) {
    const label = model.label + " · " + model.provider + (model.available ? "" : "（未配置）");
    const option = new Option(label, model.id);
    option.disabled = !model.available;
    modelOptions.add(option);
  }
  const enabled = [...modelOptions.options].filter((option) => !option.disabled);
  if (enabled.length) modelOptions.value = enabled.some((option) => option.value === previous) ? previous : enabled[0].value;
  else modelOptions.selectedIndex = modelOptions.options.length ? 0 : -1;
}

function renderControls() {
  const live = playable();
  const modelSelected = selectedMode === "model";
  byId("model-form").hidden = !modelSelected;
  const model = models.find((entry) => entry.id === modelOptions.value);
  const decision = model?.protocol === "decisions";
  byId("start-model").disabled = opening || !model?.available;
  if (decision) strategyOptions.value = "single";
  strategyOptions.disabled = decision;
  horizonOptions.disabled = strategyOptions.value === "single";
  const hint = byId("model-hint");
  if (!model?.available) {
    hint.textContent = models.some((entry) => entry.available)
      ? "当前已配置的模型只支持字符输入。"
      : "在项目根目录的 .env 或环境变量中配置 API 密钥，然后重启服务。";
  } else {
    hint.textContent = decision ? "每次从四个方向中选择一个，方向概率可在「API 输出」中查看。" : "";
  }
  hint.hidden = !hint.textContent;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === selectedMode));
    button.disabled = opening;
  });
  byId("manual-controls").hidden = selectedMode !== "manual" || run?.mode !== "manual";
  byId("auto-controls").hidden = !run || run.mode === "manual" || run.mode !== selectedMode;
  document.querySelectorAll("[data-direction]").forEach((button) => { button.disabled = !live || Boolean(flight); });
  byId("step").disabled = !live || auto || Boolean(flight);
  byId("continuous").disabled = !live || (!auto && Boolean(flight));
  byId("continuous").textContent = auto ? "暂停" : "连续运行";
  byId("end").disabled = !live;
  byId("restart").disabled = !run || opening;
}

function renderFeedback(shown) {
  const feedback = byId("feedback");
  const latest = review?.frames[review.index] ?? frames.at(-1);
  let message = "";
  let warning = false;
  if (review) {
    message = "回看 · " + (review.index ? "第 " + review.index + " 步" : "起始棋盘");
  } else if (problem) {
    message = problem;
    warning = true;
  } else if (flight?.awaitingModel) {
    message = shown?.model?.protocol === "decisions" ? "等待模型选择方向…" : "等待模型规划…";
  } else if (shown?.status && shown.status !== "running") {
    message = statusLabels[shown.status];
    warning = shown.status === "limit";
  } else if (latest?.plan?.rejection) {
    warning = true;
    message = latest.plan.rejection.reason;
  } else if (latest?.action?.outcome === "blocked") {
    warning = true;
    message = latest.action.reason;
  } else if (shown?.mode === "model" && shown.pending.length) {
    message = "本段还剩 " + shown.pending.length + " 步";
  }
  feedback.textContent = message;
  feedback.classList.toggle("warning", warning);
}

function render() {
  const shown = review?.frames[review.index]?.run ?? run;
  const level = levels.find((item) => item.id === shown?.level);
  const modeName = shown ? modes[shown.mode] + (shown.deadlockGuard ? " · 死锁预演" : "") : "";
  byId("mode-indicator").textContent = review ? "回看" : modeName;
  if (level) {
    byId("level-number").textContent = "关卡 " + level.id + " / " + levels.length;
    byId("level-title").textContent = level.name;
    byId("level-description").textContent = level.description;
    byId("filled").textContent = shown.filledGoals + " / " + shown.goals;
    for (const metric of ["moves", "pushes", "blocked"]) byId(metric).textContent = shown.metrics[metric];
    drawBoard(shown.board);
  }
  document.querySelectorAll("[data-level]").forEach((button) => {
    button.setAttribute("aria-current", String(button.dataset.level === shown?.level));
    button.disabled = opening;
  });
  renderControls();
  renderFeedback(shown);
  renderTimeline();
}

function showFrame(history, index) {
  stop();
  review = {frames: history, index};
  render();
}

function renderTimeline() {
  const history = review?.frames ?? frames;
  const index = review?.index ?? history.length - 1;
  const count = Math.max(0, history.length - 1);
  byId("trace-count").textContent = count ? count + " 条" : "";
  byId("empty-trace").hidden = count > 0;
  const timeline = byId("timeline");
  const scroll = timeline.scrollTop;
  const items = history.slice(1).map((record, offset) => {
    const position = offset + 1;
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    if (position === index) button.setAttribute("aria-current", "true");
    const stepNumber = document.createElement("span");
    stepNumber.className = "index";
    stepNumber.textContent = String(position).padStart(2, "0");
    const label = document.createElement("span");
    label.textContent = record.action ? "向" + directions[record.action.direction]
      + (record.action.planId ? " · 计划 " + record.action.planId : "") : "模型请求";
    const status = document.createElement("span");
    status.className = "action-state";
    if (!record.action || record.action.outcome === "blocked") status.classList.add("warning");
    status.textContent = record.plan?.rejection ? "预演拦截" : record.action ? outcomes[record.action.outcome] : "未执行";
    button.append(stepNumber, label, status);
    button.addEventListener("click", () => {
      showFrame(history, position);
      timeline.querySelector('[aria-current="true"]')?.focus({preventScroll: true});
    });
    row.append(button);
    return row;
  });
  timeline.replaceChildren(...items);
  timeline.scrollTop = review ? scroll : timeline.scrollHeight;
  const activeRow = review && timeline.querySelector('[aria-current="true"]');
  if (activeRow) {
    const row = activeRow.getBoundingClientRect();
    const viewport = timeline.getBoundingClientRect();
    if (row.top < viewport.top) timeline.scrollTop += row.top - viewport.top;
    if (row.bottom > viewport.bottom) timeline.scrollTop += row.bottom - viewport.bottom;
  }
  byId("replay-controls").hidden = !review;
  byId("prev-frame").disabled = !review || index <= 0;
  byId("next-frame").disabled = !review || index >= history.length - 1;
  renderTraceDetail(history, index);
}

function detailLine(detail, label, value) {
  const line = document.createElement("p");
  const strong = document.createElement("strong");
  strong.textContent = label + " ";
  line.append(strong, value);
  detail.append(line);
}

function appendApiBody(detail, label, body, api = null) {
  const disclosure = document.createElement("details");
  disclosure.className = "api-trace";
  const heading = document.createElement("summary");
  heading.textContent = label;
  disclosure.append(heading);
  if (api) {
    detailLine(disclosure, "请求地址", "POST " + api.endpoint);
    if (api.call_id) detailLine(disclosure, "调用 ID", api.call_id);
  }
  const text = document.createElement("pre");
  text.tabIndex = 0;
  text.textContent = body;
  disclosure.append(text);
  detail.append(disclosure);
}

function renderTraceDetail(history, index) {
  const detail = byId("trace-detail");
  const frame = history[index];
  detail.replaceChildren();
  detail.hidden = !frame || index === 0;
  if (detail.hidden) return;

  if (frame.action) {
    const action = frame.action;
    detailLine(detail, "实际动作", "向" + directions[action.direction] + " · " + (action.reason || outcomes[action.outcome]));
  }

  const plan = frame.plan ?? (frame.action?.planId
    ? history.find((record) => record.plan?.id === frame.action.planId)?.plan : null);
  if (!plan) return;
  const actions = plan.actions.map((direction) => directions[direction]).join(" · ");
  if (actions) detailLine(detail, "模型计划", actions);
  if (plan.note) detailLine(detail, "动作说明", plan.note);
  if (plan.error) detailLine(detail, "接口反馈", plan.error);
  if (plan.rejection) detailLine(detail, "死锁预演", plan.rejection.reason);
  if (plan.raw) detailLine(detail, "模型返回", plan.raw);
  detailLine(detail, "用时", seconds(plan.waitMs) + (plan.tokens
    ? " · " + formatter.format(plan.tokens[0] + plan.tokens[1]) + " token" : ""));
  if (plan.model !== frame.run.model?.model) detailLine(detail, "返回模型", plan.model);
  detailLine(detail, "规划时看到的棋盘", "");
  if (plan.image) {
    const image = document.createElement("img");
    image.src = plan.image;
    image.alt = "发送给模型的棋盘截图";
    detail.append(image);
  } else {
    const text = document.createElement("pre");
    text.textContent = plan.board.join("\n");
    detail.append(text);
  }
  if (plan.api) {
    const api = plan.api;
    appendApiBody(detail, "API 输入", api.request, api);
    const outputLabel = ["API 输出", api.status && "HTTP " + api.status, api.truncated && "正文已截断"]
      .filter(Boolean).join(" · ");
    appendApiBody(detail, outputLabel, api.response || "未收到响应正文。");
  }
}

function seconds(milliseconds) {
  return (milliseconds / 1000).toFixed(2) + " s";
}

function renderResults() {
  const rows = [...results.values()].reverse().map((entry) => {
    const result = entry.run;
    const isModel = result.mode === "model";
    const level = levels.find((item) => item.id === result.level);
    const metrics = result.metrics;
    const row = document.createElement("tr");
    const values = [
      level.name,
      isModel ? result.model.label + " · " + result.model.provider : modes[result.mode],
      isModel ? strategies[result.strategy] + (result.horizon > 1 ? " · K=" + result.horizon : "")
        + (result.deadlockGuard ? " · 预演拦截 " + metrics.rejectedPlans : "") : "—",
      isModel ? (result.observation === "image" ? "截图" : "字符") : "—",
      statusLabels[result.status], metrics.moves, metrics.pushes, metrics.blocked,
      isModel ? metrics.calls : "—",
      isModel ? seconds(metrics.waitMs) : "—",
      metrics.usageCalls ? formatter.format(metrics.inputTokens + metrics.outputTokens)
        + (metrics.usageCalls < metrics.calls ? "（部分）" : "") : "—",
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      if (index === 4) cell.className = result.status;
      if (index === 10 && metrics.usageCalls) {
        cell.title = "输入 " + metrics.inputTokens + " / 输出 " + metrics.outputTokens
          + "，报告 " + metrics.usageCalls + "/" + metrics.calls + " 次";
      }
      row.append(cell);
    });
    const cell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-low";
    button.textContent = "回放";
    button.setAttribute("aria-label", "回放 " + level.name + " " + values[1]);
    button.addEventListener("click", () => {
      showFrame(entry.frames, 0);
      byId("board").scrollIntoView({block: "nearest"});
    });
    cell.append(button);
    row.append(cell);
    return row;
  });
  byId("result-count").textContent = rows.length ? rows.length + " 局" : "";
  byId("empty-results").hidden = Boolean(rows.length);
  byId("result-table").hidden = !rows.length;
  byId("result-rows").replaceChildren(...rows);
}

function bindControls() {
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => changeMode(button.dataset.mode));
  });
  document.querySelectorAll("[data-direction]").forEach((button) => {
    button.addEventListener("click", () => step(button.dataset.direction));
  });
  document.addEventListener("keydown", (event) => {
    if (!keys[event.key] || !playable() || run.mode !== "manual" || event.altKey || event.ctrlKey || event.metaKey
      || event.target.matches("select, input, textarea, [contenteditable]")) return;
    event.preventDefault();
    step(keys[event.key]);
  });
  byId("model-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!run) return;
    start(run.level, {
      mode: "model",
      model: modelOptions.value,
      observation: observationOptions.value,
      strategy: strategyOptions.value,
      horizon: Number(horizonOptions.value),
      deadlockGuard: deadlockGuard.checked,
    });
  });
  observationOptions.addEventListener("change", () => {
    renderModels();
    renderControls();
  });
  modelOptions.addEventListener("change", renderControls);
  strategyOptions.addEventListener("change", renderControls);
  byId("restart").addEventListener("click", () => start(run.level, activeSettings()));
  byId("step").addEventListener("click", () => step());
  byId("continuous").addEventListener("click", setContinuous);
  byId("end").addEventListener("click", endRun);
  byId("prev-frame").addEventListener("click", () => showFrame(review.frames, review.index - 1));
  byId("next-frame").addEventListener("click", () => showFrame(review.frames, review.index + 1));
  byId("live").addEventListener("click", () => {
    review = null;
    selectedMode = run.mode;
    render();
  });
}

async function init() {
  bindControls();
  renderControls();
  try {
    const [levelData, modelData] = await Promise.all([api("/api/levels"), api("/api/models")]);
    levels = levelData.levels;
    models = modelData.models;
    renderModels();
    for (const level of levels) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.level = level.id;
      button.textContent = level.id;
      button.setAttribute("aria-label", "第 " + level.id + " 关 · " + level.name);
      button.addEventListener("click", () => start(level.id, activeSettings()));
      byId("levels").append(button);
    }
    await start(levels[0].id, {mode: "manual"});
  } catch (error) {
    problem = error.message;
    byId("level-title").textContent = "加载失败";
    render();
  }
}

init();
