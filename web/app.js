const composer = document.getElementById("composer");
const question = document.getElementById("question");
const chat = document.getElementById("chat");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const hero = document.getElementById("hero");

function setStatus(text, busy = false) {
  statusEl.innerHTML = '<span class="status-dot"></span> ' + text;
  statusEl.style.opacity = busy ? "0.65" : "1";
}

function addMessage(role, text, sources = []) {
  const wrap = document.createElement("div");
  wrap.className = "message " + role;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (role === "assistant" && sources.length) {
    const list = document.createElement("div");
    list.className = "source-list";
    sources.forEach(src => {
      const item = document.createElement("div");
      item.className = "source";
      item.innerHTML =
        '<span class="source-id">' + escapeHtml(src.source_id) + '</span>' +
        (src.document ? ' — ' + escapeHtml(src.document) : '') +
        (src.point ? '<div class="source-point">' + escapeHtml(src.point) + '</div>' : '');
      list.appendChild(item);
    });
    bubble.appendChild(list);
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
}

function removeTyping() {
  document.getElementById("typing")?.remove();
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[ch]));
}

async function ask(text) {
  const value = text.trim();
  if (!value || send.disabled) return;

  hero.style.display = "none";
  addMessage("user", value);
  question.value = "";
  send.disabled = true;
  setStatus("Ищет ответ…", true);
  addTyping();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value })
    });

    const data = await response.json();
    removeTyping();

    if (!response.ok) {
      throw new Error(data.detail || "Ошибка сервера");
    }

    addMessage("assistant", data.answer || "Ответ не получен.", data.sources || []);
    setStatus(data.success ? "Готов" : "Недостаточно данных");
  } catch (error) {
    removeTyping();
    addMessage("assistant", "Не удалось получить ответ: " + error.message);
    setStatus("Ошибка");
  } finally {
    send.disabled = false;
    question.focus();
  }
}

composer.addEventListener("submit", event => {
  event.preventDefault();
  ask(question.value);
});

question.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach(button => {
  button.addEventListener("click", () => ask(button.dataset.question));
});

setStatus("Готов");