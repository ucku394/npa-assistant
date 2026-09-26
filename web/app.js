const composer = document.getElementById("composer");
const question = document.getElementById("question");
const chat = document.getElementById("chat");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const hero = document.getElementById("hero");
const counter = document.getElementById("counter");
const newChat = document.getElementById("newChat");

const STORAGE_KEY = "npa-assistant-web-history-v1";
let history = loadHistory();

function setStatus(text, state = "ready") {
  statusEl.className = "status " + state;
  statusEl.innerHTML = '<span class="status-dot"></span> ' + escapeHtml(text);
}

function loadHistory() {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-40)));
  } catch {}
}

function clearChat() {
  history = [];
  try { localStorage.removeItem(STORAGE_KEY); } catch {}
  chat.innerHTML = "";
  hero.style.display = "";
  setStatus("Готов");
  question.value = "";
  updateCounter();
  question.focus();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[ch]));
}

function renderAnswer(text) {
  const safe = escapeHtml(text || "Ответ не получен.");
  const lines = safe.split("\n");
  const html = [];
  let listItems = [];

  const flushList = () => {
    if (!listItems.length) return;
    html.push('<ol class="answer-list">' + listItems.map(item => "<li>" + item + "</li>").join("") + "</ol>");
    listItems = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      flushList();
      continue;
    }

    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) {
      listItems.push(formatInline(numbered[1]));
      continue;
    }

    flushList();
    html.push('<p class="answer-paragraph">' + formatInline(line) + "</p>");
  }

  flushList();
  return html.join("");
}

function formatInline(value) {
  return value.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

function plainText(value) {
  return String(value ?? "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\[SOURCE:[^\]]+\]/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }

  if (button) {
    const original = button.textContent;
    button.textContent = "✓ Скопировано";
    button.classList.add("copied");
    window.setTimeout(() => {
      button.textContent = original;
      button.classList.remove("copied");
    }, 1400);
  }
}

function addMessage(role, text, sources = [], meta = "") {
  const wrap = document.createElement("div");
  wrap.className = "message " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    bubble.innerHTML = renderAnswer(text);
  } else {
    bubble.textContent = text;
  }

  if (role === "assistant") {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copyAnswer = document.createElement("button");
    copyAnswer.type = "button";
    copyAnswer.className = "message-action";
    copyAnswer.textContent = "📋 Копировать ответ";
    copyAnswer.addEventListener("click", () => copyText(plainText(text), copyAnswer));
    actions.appendChild(copyAnswer);

    if (sources.length) {
      const copySource = document.createElement("button");
      copySource.type = "button";
      copySource.className = "message-action";
      copySource.textContent = "📑 Копировать нормативную ссылку";
      const reference = sources.map(src => {
        const documentName = plainText(src.document || "НПА");
        const point = plainText(src.point || "");
        const url = plainText(src.source_url || "");
        return [documentName, point].filter(Boolean).join(" — ") + (url ? "\n" + url : "");
      }).join("\n\n");
      copySource.addEventListener("click", () => copyText(reference, copySource));
      actions.appendChild(copySource);
    }

    bubble.appendChild(actions);
  }

  if (role === "assistant" && sources.length) {
    const list = document.createElement("div");
    list.className = "source-list";
    list.innerHTML = '<div class="source-heading">Использованные источники</div>';

    sources.forEach(src => {
      const item = document.createElement("div");
      item.className = "source";

      const point = src.point ? '<div class="source-point">' + escapeHtml(src.point) + "</div>" : "";
      const link = src.source_url
        ? '<a class="source-link" href="' + escapeHtml(src.source_url) + '" target="_blank" rel="noopener noreferrer">Открыть источник ↗</a>'
        : "";

      item.innerHTML =
        '<div class="source-icon">§</div>' +
        '<div class="source-body">' +
          '<div class="source-id">' + escapeHtml(src.source_id || "") + "</div>" +
          '<div class="source-document">' + escapeHtml(src.document || "НПА") + "</div>" +
          point +
          link +
        "</div>";

      list.appendChild(item);
    });

    bubble.appendChild(list);
  }

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.id = "typing";
  wrap.innerHTML = '<div class="bubble typing">Ищу релевантные фрагменты НПА…</div>';
  chat.appendChild(wrap);

  let step = 0;
  const messages = [
    "Ищу релевантные фрагменты НПА…",
    "Проверяю нормативное основание…",
    "Формирую ответ с источниками…"
  ];

  wrap.dataset.timer = String(window.setInterval(() => {
    step = (step + 1) % messages.length;
    const bubble = wrap.querySelector(".bubble");
    if (bubble) bubble.textContent = messages[step];
  }, 1800));
}

function removeTyping() {
  const typing = document.getElementById("typing");
  if (!typing) return;
  if (typing.dataset.timer) window.clearInterval(Number(typing.dataset.timer));
  typing.remove();
}

function updateCounter() {
  counter.textContent = question.value.length + " / 4000";
}

function saveTurn(userText, answer, sources, success = true) {
  history.push({
    user: userText,
    answer,
    sources: sources || [],
    success,
    time: Date.now()
  });
  history = history.slice(-40);
  saveHistory();
}

function restoreHistory() {
  if (!history.length) return;

  hero.style.display = "none";
  history.forEach(turn => {
    addMessage("user", turn.user);
    addMessage(
      "assistant",
      turn.answer,
      turn.sources || [],
      turn.success === false ? "Ответ сформирован без достаточного нормативного контекста." : ""
    );
  });
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("health");
    setStatus("Готов");
  } catch {
    setStatus("API недоступен", "error");
  }
}

async function ask(text) {
  const value = text.trim();
  if (!value || send.disabled) return;

  hero.style.display = "none";
  addMessage("user", value);
  question.value = "";
  updateCounter();
  send.disabled = true;
  setStatus("Обрабатывает…", "busy");
  addTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value })
    });

    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error("Сервер вернул некорректный ответ.");
    }

    removeTyping();

    if (!response.ok) {
      throw new Error(data.detail || "Ошибка сервера");
    }

    const answer = data.answer || "Ответ не получен.";
    const sources = data.sources || [];

    addMessage(
      "assistant",
      answer,
      sources,
      data.success ? "" : "Недостаточно релевантного нормативного контекста."
    );
    saveTurn(value, answer, sources, Boolean(data.success));
    setStatus(data.success ? "Готов" : "Недостаточно данных");
  } catch (error) {
    removeTyping();
    const errorText = "Не удалось получить ответ. Попробуйте повторить запрос.";
    const wrap = document.createElement("div");
    wrap.className = "message assistant error-message";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = errorText;

    const actions = document.createElement("div");
    actions.className = "message-actions";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "message-action retry-action";
    retry.textContent = "↻ Повторить";
    retry.addEventListener("click", () => {
      wrap.remove();
      ask(value);
    });
    actions.appendChild(retry);
    bubble.appendChild(actions);
    wrap.appendChild(bubble);
    chat.appendChild(wrap);
    wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setStatus("Ошибка", "error");
  } finally {
    send.disabled = false;
    question.focus();
  }
}

composer.addEventListener("submit", event => {
  event.preventDefault();
  ask(question.value);
});

question.addEventListener("input", updateCounter);

question.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

newChat.addEventListener("click", clearChat);

document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

updateCounter();
restoreHistory();
checkHealth();
question.focus();
