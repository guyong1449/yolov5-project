const state = {
  taskDefinitions: null,
  session: {},
  currentTask: "train",
  runtime: null,
  forms: {},
  eventSource: null,
  commandHistory: [],
};

let baseCommand = "";

const EXAMPLE_COMMANDS = {
  train: [
    {
      label: "标准训练 (SGD, 70 epochs)",
      command:
        'D:/Miniconda3/python.exe scripts/run_with_log.py -- \\\n  D:/Miniconda3/python.exe train.py \\\n  --data F:/1/labelimg/data/test1_stride10/data.yaml \\\n  --weights F:\\1\\yolov5-master\\runs\\train\\test1_stride10_sgd_70e3\\weights\\best.pt \\\n  --epochs 70 --batch-size 4 --imgsz 640 \\\n  --device 0 --seed 0 --workers 2 --patience 20 \\\n  --optimizer SGD --project runs/train --name test1_stride10_sgd_70e',
    },
  ],
  detect: [
    {
      label: "VOC 抽帧导出 (vid-stride=10)",
      command:
        'D:/Miniconda3/python.exe detect.py \\\n  --weights checkpoint/yolov5_best.pt \\\n  --source "F:/1/video/output" \\\n  --data F:/1/labelimg/data/test1_stride10/data.yaml \\\n  --imgsz 640 --device 0 \\\n  --project runs/detect --name voc_stride10 \\\n  --voc-root F:/1/labelimg/data/test1_stride10 \\\n  --vid-stride 10 --save-img-frames --nosave --incremental-mp4',
    },
    {
      label: "封装脚本 (extract_voc_stride10)",
      command:
        'D:/Miniconda3/python.exe scripts/extract_voc_stride10.py \\\n  --weights checkpoint/yolov5_best.pt \\\n  --source "F:/1/video/output" \\\n  --voc-root F:/1/labelimg/data/test2_stride10 \\\n  --data-yaml F:/1/labelimg/data/test1_stride10/data.yaml \\\n  --device 0',
    },
  ],
  val: [],
  fiftyone: [
    {
      label: "FiftyOne 连续去重",
      command:
        'D:\\Miniconda3\\envs\\f312\\python.exe tools\\fiftyone\\fiftyone_run_full_dedup_pipeline.py \\\n  --dataset-name test1_stride10_voc \\\n  --model clip-vit-base32-torch \\\n  --brain-key clip_vit_base32_sim \\\n  --approx-threshold 0.12 --approx-group-keep-ratio 0.3 \\\n  --voc-root "F:\\1\\labelimg\\data\\test1_stride10\\fiftyone_voc" \\\n  --export-dir "F:\\1\\labelimg\\data\\test1_stride10\\fiftyone_voc_deduped" \\\n  --report-dir "F:\\1\\labelimg\\data\\test1_stride10\\fiftyone_voc\\dedup_reports" \\\n  --overwrite',
    },
  ],
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = Array.isArray(payload.detail)
        ? payload.detail.join("\n")
        : payload.detail || detail;
    } catch (error) {
      detail = response.statusText;
    }
    throw new Error(detail);
  }
  return response.json();
}

async function init() {
  const payload = await fetchJson("/api/task-definitions");
  state.taskDefinitions = payload.tasks;
  state.session = payload.session || {};
  state.commandHistory = payload.command_history || [];
  for (const [taskType, spec] of Object.entries(state.taskDefinitions)) {
    state.forms[taskType] = {
      ...spec.defaults,
      ...(state.session.recent_values?.[taskType] || {}),
    };
  }
  await refreshRuntime();
  renderTaskCards();
  renderTaskForm();
  connectLogs();
  setInterval(refreshRuntime, 2500);
  const copyLogPathBtn = $("copy-log-path-btn");
  if (copyLogPathBtn) copyLogPathBtn.addEventListener("click", () =>
    copyText(state.runtime?.last_log_path || ""));
  const copyLogBtn = $("copy-log-btn");
  if (copyLogBtn) copyLogBtn.addEventListener("click", () =>
    copyText(($("log-output")?.textContent) || ""));
  const clearLogBtn = $("clear-log-btn");
  if (clearLogBtn) clearLogBtn.addEventListener("click", async () => {
    await fetchJson("/api/logs/clear", { method: "POST" });
    const logOutput = $("log-output");
    if (logOutput) logOutput.textContent = "";
  });
}

async function refreshRuntime() {
  state.runtime = await fetchJson("/api/runtime-state");
  if (state.runtime?.command_history) {
    state.commandHistory = state.runtime.command_history;
  }
  if (state.runtime?.recent_logs) {
    const logOutput = $("log-output");
    if (logOutput) logOutput.textContent =
      `${state.runtime.recent_logs.join("\n")}${state.runtime.recent_logs.length ? "\n" : ""}`;
  }
  renderTaskCards();
  updateRuntimeChrome();
}

function renderTaskCards() {
  if (!state.taskDefinitions) return;
  const root = $("task-cards");
  if (!root) return;
  root.innerHTML = "";
  for (const [taskType, spec] of Object.entries(state.taskDefinitions)) {
    const card = document.createElement("section");
    card.className = `task-card ${state.currentTask === taskType ? "active" : ""}`;
    const isActiveRuntime = state.runtime?.active_task === taskType;
    const recentOutput =
      state.runtime?.recent_outputs?.[taskType] ||
      state.session.recent_outputs?.[taskType] ||
      "未记录";
    const recentLog =
      state.runtime?.recent_logs_paths?.[taskType] ||
      state.session.recent_logs?.[taskType] ||
      "未记录";
    card.innerHTML = `
      <div class="task-card__header">
        <h3>${spec.display_name}</h3>
        <span class="task-card__status">${isActiveRuntime ? state.runtime.status : "idle"}</span>
      </div>
      <small class="task-card__line"><span>输出</span><code>${recentOutput}</code></small>
      <small class="task-card__line"><span>日志</span><code>${recentLog}</code></small>
      <div class="task-card__actions">
        <button type="button" class="secondary card-copy-btn" data-copy="${escapeHtml(recentOutput)}">复制输出</button>
        <button type="button" class="secondary card-copy-btn" data-copy="${escapeHtml(recentLog)}">复制日志</button>
      </div>
    `;
    card.addEventListener("click", () => {
      state.currentTask = taskType;
      renderTaskCards();
      renderTaskForm();
    });
    card.querySelectorAll(".card-copy-btn").forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        copyText(event.currentTarget.dataset.copy || "");
      });
    });
    root.appendChild(card);
  }
}

function $(id) {
  return document.getElementById(id);
}

function updateRuntimeChrome() {
  const taskStatus = $("task-status");
  if (taskStatus) taskStatus.textContent = state.runtime?.status || "idle";
  const logMeta = $("log-meta");
  if (logMeta) logMeta.textContent = state.runtime?.last_log_path || "等待任务启动";

  const pidEl = $("task-pid");
  if (pidEl) {
    const pid = state.runtime?.pid;
    const activeTask = state.runtime?.active_task;
    const status = state.runtime?.status || "idle";
    if (pid && ["running", "stopping"].includes(status)) {
      const wrapper = activeTask === "fiftyone" ? "direct" : "run_with_log.py";
      pidEl.textContent = `PID: ${pid}  |  wrapper: ${wrapper}`;
    } else if (pid && status !== "idle") {
      const wrapper = activeTask === "fiftyone" ? "direct" : "run_with_log.py";
      pidEl.textContent = `PID: ${pid}  |  wrapper: ${wrapper}  |  ${status}`;
    } else {
      pidEl.textContent = "";
    }
  }

  const isRunning = ["running", "stopping"].includes(state.runtime?.status || "");
  const sameTask = state.runtime?.active_task === state.currentTask;
  const startBtn = $("start-btn");
  const stopBtn = $("stop-btn");
  if (startBtn) startBtn.disabled = isRunning && !sameTask;
  if (stopBtn) stopBtn.disabled = !isRunning || !sameTask;
}

function renderTaskForm() {
  const spec = state.taskDefinitions[state.currentTask];
  const taskTitle = $("task-title");
  if (taskTitle) taskTitle.textContent = spec.display_name;
  const taskDesc = $("task-description");
  if (taskDesc) taskDesc.textContent = spec.description;
  const root = $("task-form");
  if (!root) return;
  root.innerHTML = "";

  const BASIC_OUTPUT_IDS = ["basic", "output"];
  const sideGroups = spec.field_groups.filter((g) =>
    BASIC_OUTPUT_IDS.includes(g.id)
  );
  const otherGroups = spec.field_groups.filter(
    (g) => !BASIC_OUTPUT_IDS.includes(g.id)
  );

  if (sideGroups.length > 0) {
    const row = document.createElement("div");
    row.className = "field-group-row";
    for (const group of sideGroups) {
      row.appendChild(buildFieldGroup(group, spec));
    }
    root.appendChild(row);
  }

  for (const group of otherGroups) {
    root.appendChild(buildFieldGroup(group, spec));
  }

  const commandBox = document.createElement("section");
  commandBox.className = "field-group";
  const examples = EXAMPLE_COMMANDS[state.currentTask] || [];
  const examplesHtml =
    examples.length > 0
      ? `<details class="examples-block">
           <summary>示例命令 (来自 docs/run-commands.md)</summary>
           ${examples
             .map(
               (ex) => `
             <div class="example-item">
               <div class="example-item__label">${ex.label}</div>
               <pre class="copyable-text">${escapeHtml(ex.command)}</pre>
               <button type="button" class="secondary example-item__copy copy-cmd-btn" data-copy="${escapeHtml(ex.command)}">复制此示例</button>
             </div>
           `
             )
             .join("")}
         </details>`
      : "";

  const historyItems = state.commandHistory || [];
  const historyHtml =
    historyItems.length > 0
      ? `<details class="examples-block" id="command-history-block">
           <summary>最近使用的命令 (${historyItems.length})</summary>
           ${historyItems
             .map(
               (entry, idx) => `
             <div class="example-item">
               <div class="example-item__label">${escapeHtml(entry.task_type)} &middot; ${new Date(entry.timestamp * 1000).toLocaleString()}</div>
               <pre class="copyable-text">${escapeHtml(entry.command)}</pre>
               <button type="button" class="secondary example-item__copy use-history-btn" data-command="${escapeHtml(entry.command)}">使用此命令</button>
             </div>
           `
             )
             .join("")}
         </details>`
      : "";

  commandBox.innerHTML = `
    <div class="section-header">
      <h3>命令预览（可直接编辑追加参数）</h3>
      <div class="field-actions">
        <button id="copy-command-btn" type="button" class="secondary">复制命令</button>
      </div>
    </div>
    <textarea id="command-preview" class="command-preview" spellcheck="false">生成中...</textarea>
    ${historyHtml}
    ${examplesHtml}
  `;
  root.appendChild(commandBox);

  const actions = document.createElement("section");
  actions.className = "field-group";
  actions.innerHTML = `
    <h3>操作</h3>
    <div class="primary-actions">
      <button id="validate-btn" type="button">校验</button>
      <button id="start-btn" type="button">启动</button>
      <button id="stop-btn" type="button" class="warn">停止</button>
      <button id="reset-btn" type="button" class="secondary">重置当前表单</button>
      <button id="restore-btn" type="button" class="secondary">恢复最近配置</button>
      <button id="open-output-btn" type="button" class="secondary">打开输出目录</button>
      <button id="open-log-btn" type="button" class="secondary">打开日志目录</button>
    </div>
    <p id="action-feedback" class="muted copyable-text"></p>
  `;
  root.appendChild(actions);

  bindActionButtons();
  refreshCommandPreview();
  const copyCmdBtn = $("copy-command-btn");
  if (copyCmdBtn) copyCmdBtn.addEventListener("click", () => {
    copyText(($("command-preview")?.value) || "");
  });
  document.querySelectorAll(".copy-cmd-btn").forEach((btn) => {
    btn.addEventListener("click", () => copyText(btn.dataset.copy || ""));
  });
  document.querySelectorAll(".use-history-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const textarea = $("command-preview");
      if (textarea) {
        textarea.value = btn.dataset.command || "";
        handleCommandEdit();
      }
    });
  });
}

function buildFieldGroup(group, spec) {
  const groupEl = document.createElement("section");
  groupEl.className = "field-group";

  const fields = spec.fields
    .filter((field) => field.group === group.id)
    .filter((field) => shouldShowField(field));

  const title = document.createElement("h3");
  title.textContent = group.label;
  groupEl.appendChild(title);

  if (fields.length > 0) {
    const grid = document.createElement("div");
    grid.className = "fields-grid";
    fields.forEach((field) => grid.appendChild(buildField(field)));
    groupEl.appendChild(grid);
  }

  return groupEl;
}

function shouldShowField(field) {
  const visibleWhen = field.visible_when || {};
  return Object.entries(visibleWhen).every(
    ([name, value]) => state.forms[state.currentTask][name] === value
  );
}

function buildField(field) {
  const container = document.createElement("div");
  container.className = `field ${field.kind === "textarea" ? "wide" : ""}`;
  const value = state.forms[state.currentTask][field.name] ?? "";
  const helpTitle = field.help_text ? ` title="${escapeHtml(field.help_text)}"` : "";
  if (field.kind === "checkbox") {
    container.innerHTML = `
      <label${helpTitle}>${field.label}</label>
      <div class="checkbox-row">
        <input type="checkbox" ${value ? "checked" : ""} data-field="${field.name}" />
        <span>${field.help_text || field.label}</span>
      </div>
    `;
  } else if (field.kind === "select") {
    const options = field.choices
      .map(
        (option) =>
          `<option value="${option.value}" ${option.value === value ? "selected" : ""}>${option.label}</option>`
      )
      .join("");
    container.innerHTML = `<label${helpTitle}>${field.label}</label><select data-field="${field.name}">${options}</select>`;
  } else if (field.kind === "textarea") {
    container.innerHTML = `<label${helpTitle}>${field.label}</label><textarea data-field="${field.name}" placeholder="${field.placeholder || ""}">${value}</textarea>`;
  } else {
    const type = field.kind === "number" ? "number" : "text";
    const stepAttr = field.step ? ` step="${field.step}"` : "";
    container.innerHTML = `
      <label${helpTitle}>${field.label}</label>
      <div class="field-actions">
        <input type="${type}" data-field="${field.name}" value="${value}" placeholder="${field.placeholder || ""}"${stepAttr} />
        ${field.browsable
          ? `<button type="button" class="secondary browse-btn" data-kind="${field.browse_kind}" data-field="${field.name}">浏览</button>`
          : ""}
      </div>
    `;
  }
  container.querySelectorAll("[data-field]").forEach((input) => {
    input.addEventListener("change", handleFieldChange);
  });
  container.querySelectorAll(".browse-btn").forEach((button) => {
    button.addEventListener("click", handleBrowse);
  });
  return container;
}

function handleFieldChange(event) {
  const fieldName = event.target.dataset.field;
  state.forms[state.currentTask][fieldName] =
    event.target.type === "checkbox" ? event.target.checked : event.target.value;
  renderTaskForm();
}

async function handleBrowse(event) {
  const fieldName = event.target.dataset.field;
  const kind =
    event.target.dataset.kind === "directory" ? "directory" : "file";
  const result = await fetchJson("/api/dialog/select-path", {
    method: "POST",
    body: JSON.stringify({
      kind,
      title: `选择 ${fieldName}`,
      initial_path: state.forms[state.currentTask][fieldName] || "",
    }),
  });
  if (result.path) {
    state.forms[state.currentTask][fieldName] = result.path;
    renderTaskForm();
  }
}

async function refreshCommandPreview() {
  const preview = await fetchJson(`/api/tasks/${state.currentTask}/preview`, {
    method: "POST",
    body: JSON.stringify({
      task_type: state.currentTask,
      values: state.forms[state.currentTask],
    }),
  });
  baseCommand = preview.command;
  const textarea = $("command-preview");
  if (textarea) {
    const extraArgs = state.forms[state.currentTask].extra_args || "";
    textarea.value = extraArgs ? baseCommand + " " + extraArgs : baseCommand;
    textarea.addEventListener("input", handleCommandEdit);
  }
}

function normalizeCommand(cmd) {
  return cmd.replace(/\\\n\s*/g, " ").replace(/\s+/g, " ").trim();
}

function handleCommandEdit() {
  const textarea = $("command-preview");
  if (!textarea || !baseCommand) return;
  const editedNorm = normalizeCommand(textarea.value);
  const baseNorm = normalizeCommand(baseCommand);
  if (editedNorm.startsWith(baseNorm)) {
    state.forms[state.currentTask].extra_args = editedNorm.slice(baseNorm.length).trim();
  } else {
    state.forms[state.currentTask].extra_args = textarea.value;
  }
}

async function copyText(text) {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const feedback = $("action-feedback");
    if (feedback) feedback.textContent = "已复制";
  } catch (error) {
    const feedback = $("action-feedback");
    if (feedback) feedback.textContent = `复制失败：${error.message}`;
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function bindActionButtons() {
  const validateBtn = $("validate-btn");
  if (validateBtn) validateBtn.addEventListener("click", validateCurrentTask);
  const startBtn = $("start-btn");
  if (startBtn) startBtn.addEventListener("click", startCurrentTask);
  const stopBtn = $("stop-btn");
  if (stopBtn) stopBtn.addEventListener("click", stopCurrentTask);
  const resetBtn = $("reset-btn");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    state.forms[state.currentTask] = {
      ...state.taskDefinitions[state.currentTask].defaults,
    };
    renderTaskForm();
  });
  const restoreBtn = $("restore-btn");
  if (restoreBtn) restoreBtn.addEventListener("click", () => {
    state.forms[state.currentTask] = {
      ...state.taskDefinitions[state.currentTask].defaults,
      ...(state.session.recent_values?.[state.currentTask] || {}),
    };
    renderTaskForm();
  });
  const openOutputBtn = $("open-output-btn");
  if (openOutputBtn) openOutputBtn.addEventListener("click", () => openPath(state.runtime?.last_output_path));
  const openLogBtn = $("open-log-btn");
  if (openLogBtn) openLogBtn.addEventListener("click", () => openPath(state.runtime?.last_log_path));
}

async function validateCurrentTask() {
  const feedback = $("action-feedback");
  try {
    const result = await fetchJson(
      `/api/tasks/${state.currentTask}/validate`,
      {
        method: "POST",
        body: JSON.stringify({
          task_type: state.currentTask,
          values: state.forms[state.currentTask],
        }),
      }
    );
    state.session.recent_values = state.session.recent_values || {};
    state.session.recent_values[state.currentTask] = result.normalized_values;
    if (feedback) feedback.textContent = result.ok
      ? "校验通过"
      : result.errors.join("；");
    const cmdPreview = $("command-preview");
    if (cmdPreview) {
      baseCommand = result.command;
      const extraArgs = state.forms[state.currentTask].extra_args || "";
      cmdPreview.value = extraArgs ? result.command + " " + extraArgs : result.command;
    }
  } catch (error) {
    if (feedback) feedback.textContent = error.message;
  }
}

async function startCurrentTask() {
  const feedback = $("action-feedback");
  try {
    const result = await fetchJson(`/api/tasks/${state.currentTask}/start`, {
      method: "POST",
      body: JSON.stringify({
        task_type: state.currentTask,
        values: state.forms[state.currentTask],
      }),
    });
    if (feedback) feedback.textContent = `已启动 ${result.active_task}`;
    await refreshRuntime();
  } catch (error) {
    if (feedback) feedback.textContent = error.message;
  }
}

async function stopCurrentTask() {
  const feedback = $("action-feedback");
  try {
    await fetchJson("/api/tasks/stop", { method: "POST" });
    if (feedback) feedback.textContent = "已发送停止请求";
    await refreshRuntime();
  } catch (error) {
    if (feedback) feedback.textContent = error.message;
  }
}

async function openPath(pathValue) {
  if (!pathValue) return;
  try {
    await fetchJson("/api/open-path", {
      method: "POST",
      body: JSON.stringify({ path: pathValue }),
    });
  } catch (error) {
    const feedback = $("action-feedback");
    if (feedback) feedback.textContent = error.message;
  }
}

function connectLogs() {
  const logOutput = $("log-output");
  if (!logOutput) return;
  state.eventSource = new EventSource("/api/logs/stream");
  state.eventSource.onmessage = (event) => {
    if (!event.data) return;
    logOutput.textContent += `${event.data}\n`;
    logOutput.scrollTop = logOutput.scrollHeight;
  };
}

init().catch((error) => {
  const taskForm = $("task-form");
  if (taskForm) taskForm.innerHTML =
    `<div class="field-group"><h3>初始化失败</h3><p>${error.message}</p></div>`;
});
