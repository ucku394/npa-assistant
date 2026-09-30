const photoInput = document.getElementById("photoInput");
const inspectionResult = document.getElementById("inspectionResult");
let lastInspectionFindings = [];

function renderInspection(data) {
  lastInspectionFindings = data.findings || [];
  const confirmed = lastInspectionFindings.filter(x => x.status === "confirmed" && (x.legal_basis || []).length);
  const potential = lastInspectionFindings.filter(x => x.status === "potential");
  const observations = data.observations || [];
  let html = '<div class="inspection-scene"><strong>Область:</strong> ' + escapeHtml(data.category || "Требует определения") + '</div>';
  if (data.scene) html += '<div class="inspection-scene"><strong>Что видно:</strong> ' + escapeHtml(data.scene) + '</div>';
  if (observations.length) html += '<div class="inspection-title">👁️ Наблюдения</div><ul>' + observations.map(x => '<li>' + escapeHtml(x.description || "") + '</li>').join("") + '</ul>';
  if (confirmed.length) {
    html += '<div class="inspection-title confirmed">🚨 Подтверждённые нарушения</div>' +
      confirmed.map((x,i) => '<div class="finding confirmed"><strong>' + (i+1) + '. ' + escapeHtml(x.violation || x.description || "Нарушение") + '</strong><div>НПА: ' + escapeHtml((x.legal_basis || []).map(b => [b.document,b.point].filter(Boolean).join(", ")).join("; ")) + '</div><div>Действие: ' + escapeHtml(x.corrective_action || "") + '</div></div>').join("") +
      '<button type="button" class="prescription-button" id="makePrescription">📄 Сформировать проект предписания</button>';
  }
  if (potential.length) html += '<div class="inspection-title potential">⚠️ Требуют проверки</div>' + potential.map(x => '<div class="finding"><strong>' + escapeHtml(x.violation || x.description || "") + '</strong><div>' + escapeHtml((x.verification_needed || []).join(" ")) + '</div></div>').join("");
  if (!confirmed.length) html += '<div class="inspection-note">Проект предписания доступен только при наличии подтверждённых нарушений с конкретным нормативным основанием.</div>';
  inspectionResult.innerHTML = html; inspectionResult.hidden = false;
  const button = document.getElementById("makePrescription"); if (button) button.addEventListener("click", createPrescription);
}

async function createPrescription() {
  const button = document.getElementById("makePrescription"); if (button) {button.disabled=true;button.textContent="Формирую DOCX…";}
  try {
    const response = await fetch("/api/prescription",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({findings:lastInspectionFindings})});
    if (!response.ok) { let msg="Не удалось сформировать предписание."; try {const d=await response.json();msg=d.detail||msg;} catch {} throw new Error(msg); }
    const blob=await response.blob(), url=URL.createObjectURL(blob), a=document.createElement("a");
    a.href=url;a.download="proekt_predpisaniya.docx";document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
  } catch(e) { alert(e.message || "Ошибка формирования предписания."); }
  finally { if(button){button.disabled=false;button.textContent="📄 Сформировать проект предписания";} }
}

if (photoInput) photoInput.addEventListener("change", async () => {
  const file=photoInput.files && photoInput.files[0]; if(!file)return;
  inspectionResult.hidden=false; inspectionResult.innerHTML='<div class="inspection-loading">📷 Анализирую фотографию и проверяю признаки по НПА…</div>';
  try {
    const form=new FormData(); form.append("file",file);
    const response=await fetch("/api/inspect",{method:"POST",body:form}), data=await response.json();
    if(!response.ok)throw new Error(data.detail||"Ошибка фотоинспекции."); renderInspection(data);
  } catch(e) {inspectionResult.innerHTML='<div class="inspection-error">'+escapeHtml(e.message||"Не удалось выполнить фотоинспекцию.")+'</div>';}
  finally {photoInput.value="";}
});

const composer = document.getElementById("composer");
const question = document.getElementById("question");
const chat = document.getElementById("chat");
const send = document.getElementById("send");
const statusEl = document.getElementById("status");
const hero = document.getElementById("hero");
const counter = document.getElementById("counter");
const newChat = document.getElementById("newChat");
const historyPanel = document.getElementById("historyPanel");
const historyList = document.getElementById("historyList");
const clearHistoryButton = document.getElementById("clearHistory");

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
  renderHistory();
  question.focus();
}

function renderHistory() {
  if (!historyPanel || !historyList) return;

  historyList.innerHTML = "";
  if (!history.length) {
    historyPanel.classList.remove("has-history");
    return;
  }

  historyPanel.classList.add("has-history");

  history.slice().reverse().forEach((turn, reverseIndex) => {
    const index = history.length - 1 - reverseIndex;
    const item = document.createElement("div");
    item.className = "history-item";

    const body = document.createElement("div");
    body.className = "history-body";

    const text = document.createElement("div");
    text.className = "history-question";
    text.textContent = turn.user || "Без вопроса";

    const meta = document.createElement("div");
    meta.className = "history-meta";
    const date = turn.time ? new Date(turn.time) : null;
    meta.textContent = date && !Number.isNaN(date.getTime())
      ? date.toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })
      : "Ранее";
    if (turn.sources && turn.sources.length) {
      meta.textContent += " · " + turn.sources.length + " " +
        (turn.sources.length === 1 ? "источник" : "источника");
    }

    body.appendChild(text);
    body.appendChild(meta);

    const repeat = document.createElement("button");
    repeat.type = "button";
    repeat.className = "history-repeat";
    repeat.textContent = "Повторить";
    repeat.title = "Повторить этот вопрос";
    repeat.addEventListener("click", () => {
      question.value = turn.user || "";
      updateCounter();
      question.focus();
      question.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    item.appendChild(body);
    item.appendChild(repeat);
    historyList.appendChild(item);
  });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;"
  }[ch]));
}

function renderAnswer(text, citationCount = 0, sources = []) {
  let safe = escapeHtml(text || "Ответ не получен.");
  if (citationCount > 0) {
    safe = safe.replace(/\[(\d+)\]/g, (match, number) => {
      const n = Number(number);
      if (n < 1 || n > citationCount) return match;

      const source = sources[n - 1] || {};
      const documentName = escapeHtml(source.document || "Нормативный источник");
      const point = escapeHtml(source.point || "");
      const shortTitle = point ? documentName + " · " + point : documentName;

      return '<button type="button" class="answer-citation" data-citation="' + n +
        '" aria-label="Источник ' + n + ': ' + documentName +
        (point ? ', ' + point : '') +
        '" title="' + shortTitle + '">[' + n + ']</button>';
    });
  }
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
    .replace(/\[(\d+)\]/g, "")
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

function formatCount(value, one, few, many) {
  const n = Math.abs(Number(value) || 0);
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
  return many;
}

function addSourceSummary(container, sources, rag = null) {
  if (!sources.length) return;

  const summary = document.createElement("div");
  summary.className = "source-summary";

  const usedCount = sources.length;
  const usedLabel = formatCount(
    usedCount,
    "источник",
    "источника",
    "источников"
  );

  let details = `Использовано нормативных источников: <strong>${usedCount}</strong> ${usedLabel}`;

  const finalCount = Number(rag?.final_count || 0);
  if (finalCount > 0) {
    const fragmentLabel = formatCount(
      finalCount,
      "фрагмент",
      "фрагмента",
      "фрагментов"
    );
    details += `<span class="source-summary-separator">·</span> Отобрано RAG: <strong>${finalCount}</strong> ${fragmentLabel}`;
  }

  summary.innerHTML =
    '<span class="source-summary-icon">§</span>' +
    '<span class="source-summary-text">' + details + "</span>";

  container.appendChild(summary);
}

function addMessage(role, text, sources = [], meta = "", rag = null, citationText = null) {
  const wrap = document.createElement("div");
  wrap.className = "message " + role;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    bubble.innerHTML = renderAnswer(citationText || text, sources.length, sources);
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
    bubble.addEventListener("click", event => {
      const citation = event.target.closest(".answer-citation");
      if (!citation) return;

      const citationNumber = Number(citation.dataset.citation || 0);
      if (!citationNumber) return;

      const source = sources[citationNumber - 1] || {};
      const sourceContent = bubble.querySelector(".source-content");
      const sourceCards = bubble.querySelectorAll(".source-card");
      const target = sourceCards[citationNumber - 1];
      const toggle = bubble.querySelector(".source-toggle");

      if (!target) return;

      const existingPopover = bubble.querySelector(".citation-popover");
      const sameCitation = existingPopover &&
        existingPopover.dataset.citation === String(citationNumber);

      if (sameCitation) {
        existingPopover.remove();
        sourceContent.hidden = false;
        if (toggle) {
          toggle.setAttribute("aria-expanded", "true");
          toggle.classList.add("expanded");
          const toggleText = toggle.querySelector(".source-toggle-text");
          const toggleIcon = toggle.querySelector(".source-toggle-icon");
          if (toggleText) toggleText.textContent = "Скрыть источники";
          if (toggleIcon) toggleIcon.textContent = "⌃";
        }
        target.classList.add("source-card-highlight");
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        window.setTimeout(() => target.classList.remove("source-card-highlight"), 1800);
        return;
      }

      if (existingPopover) existingPopover.remove();

      const popover = document.createElement("div");
      popover.className = "citation-popover";
      popover.dataset.citation = String(citationNumber);

      const title = document.createElement("div");
      title.className = "citation-popover-title";
      title.textContent = "Источник " + citationNumber;
      popover.appendChild(title);

      const documentName = document.createElement("div");
      documentName.className = "citation-popover-document";
      documentName.textContent = source.document || "Нормативный источник";
      popover.appendChild(documentName);

      if (source.point) {
        const point = document.createElement("div");
        point.className = "citation-popover-point";
        point.textContent = source.point;
        popover.appendChild(point);
      }

      const hint = document.createElement("div");
      hint.className = "citation-popover-hint";
      hint.textContent = "Нажмите ещё раз, чтобы открыть источник";
      popover.appendChild(hint);

      citation.insertAdjacentElement("afterend", popover);
    });

    const list = document.createElement("div");
    list.className = "source-list";

    const sourceToggle = document.createElement("button");
    sourceToggle.type = "button";
    sourceToggle.className = "source-toggle";
    sourceToggle.setAttribute("aria-expanded", "false");

    const sourceToggleText = document.createElement("span");
    sourceToggleText.className = "source-toggle-text";

    const sourceToggleIcon = document.createElement("span");
    sourceToggleIcon.className = "source-toggle-icon";
    sourceToggleIcon.textContent = "⌄";

    sourceToggle.appendChild(sourceToggleText);
    sourceToggle.appendChild(sourceToggleIcon);

    const sourceContent = document.createElement("div");
    sourceContent.className = "source-content";
    sourceContent.hidden = true;

    addSourceSummary(sourceContent, sources, rag);

    const heading = document.createElement("div");
    heading.className = "source-heading";
    heading.textContent = "Использованные источники";
    sourceContent.appendChild(heading);

    const sourceCards = document.createElement("div");
    sourceCards.className = "source-cards";
    sourceContent.appendChild(sourceCards);

    const updateSourceToggle = (expanded) => {
      sourceToggle.setAttribute("aria-expanded", String(expanded));
      sourceToggle.classList.toggle("expanded", expanded);
      sourceToggleIcon.textContent = expanded ? "⌃" : "⌄";
      sourceToggleText.textContent = expanded
        ? "Скрыть источники"
        : "Показать источники (" + sources.length + ")";
    };

    sourceToggle.addEventListener("click", () => {
      const expanded = sourceToggle.getAttribute("aria-expanded") === "true";
      sourceContent.hidden = expanded;
      updateSourceToggle(!expanded);
    });

    updateSourceToggle(false);
    list.appendChild(sourceToggle);
    list.appendChild(sourceContent);

    sources.forEach((src, index) => {
      const item = document.createElement("article");
      item.className = "source-card";

      const documentName = escapeHtml(src.document || "Нормативный правовой акт");
      const point = src.point
        ? '<div class="source-point">' + escapeHtml(src.point) + "</div>"
        : "";
      const sourceId = src.source_id
        ? '<div class="source-id">' + escapeHtml(src.source_id) + "</div>"
        : "";
      const citationOrder = Number(src.citation_order || 0);
      const citation = citationOrder > 0
        ? '<span class="source-citation">Цитирование в ответе · ' + citationOrder + "</span>"
        : "";
      const link = src.source_url
        ? '<a class="source-link" href="' + escapeHtml(src.source_url) + '" target="_blank" rel="noopener noreferrer">Открыть первоисточник <span>↗</span></a>'
        : '<span class="source-unavailable">Ссылка на первоисточник не указана</span>';

      item.innerHTML =
        '<div class="source-card-top">' +
          '<span class="source-number">Источник ' + (index + 1) + "</span>" +
          citation +
          sourceId +
        "</div>" +
        '<div class="source-document">' + documentName + "</div>" +
        point +
        '<div class="source-card-bottom">' + link + "</div>";

      sourceCards.appendChild(item);
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

function addSearchProgress() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";
  wrap.id = "search-progress";

  const bubble = document.createElement("div");
  bubble.className = "bubble search-progress";
  bubble.innerHTML = '<div class="progress-title">Обрабатываю вопрос</div><div class="progress-steps"></div>';

  const steps = [
    "Поиск релевантных НПА",
    "Анализ найденных положений",
    "Проверка нормативных источников",
    "Формирование ответа"
  ];

  const list = bubble.querySelector(".progress-steps");
  steps.forEach((label, index) => {
    const step = document.createElement("div");
    step.className = "progress-step" + (index === 0 ? " active" : "");
    step.dataset.index = String(index);
    step.innerHTML =
      '<span class="progress-icon">○</span>' +
      '<span class="progress-label">' + escapeHtml(label) + "</span>";
    list.appendChild(step);
  });

  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });

  let current = 0;
  const timer = window.setInterval(() => {
    current += 1;
    if (current >= steps.length) {
      current = steps.length - 1;
      window.clearInterval(timer);
    }

    list.querySelectorAll(".progress-step").forEach((step, index) => {
      step.classList.toggle("done", index < current);
      step.classList.toggle("active", index === current);
      const icon = step.querySelector(".progress-icon");
      if (icon) icon.textContent = index < current ? "✓" : index === current ? "●" : "○";
    });
  }, 1200);

  wrap.dataset.timer = String(timer);
}

function finishSearchProgress(success = true) {
  const wrap = document.getElementById("search-progress");
  if (!wrap) return;

  if (wrap.dataset.timer) window.clearInterval(Number(wrap.dataset.timer));

  wrap.querySelectorAll(".progress-step").forEach(step => {
    step.classList.remove("active");
    step.classList.add("done");
    const icon = step.querySelector(".progress-icon");
    if (icon) icon.textContent = "✓";
  });

  const title = wrap.querySelector(".progress-title");
  if (title) title.textContent = success ? "Ответ готов" : "Поиск завершён";

  window.setTimeout(() => wrap.remove(), 260);
}

function removeSearchProgress() {
  const wrap = document.getElementById("search-progress");
  if (!wrap) return;
  if (wrap.dataset.timer) window.clearInterval(Number(wrap.dataset.timer));
  wrap.remove();
}

function updateCounter() {
  counter.textContent = question.value.length + " / 4000";
}

function saveTurn(userText, answer, sources, success = true, rag = null, answerWithCitations = null) {
  history.push({
    user: userText,
    answer,
    answerWithCitations: answerWithCitations || answer,
    sources: sources || [],
    success,
    rag: rag || null,
    time: Date.now()
  });
  history = history.slice(-40);
  saveHistory();
  renderHistory();
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
      turn.success === false ? "Ответ сформирован без достаточного нормативного контекста." : "",
      turn.rag || null,
      turn.answerWithCitations || turn.answer || ""
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
  addSearchProgress();

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

    if (!response.ok) {
      throw new Error(data.detail || "Ошибка сервера");
    }

    const answer = data.answer || "Ответ не получен.";
    const sources = data.sources || [];

    finishSearchProgress(Boolean(data.success));
    addMessage(
      "assistant",
      answer,
      sources,
      data.success ? "" : "Недостаточно релевантного нормативного контекста.",
      data.rag || null,
      data.answer_with_citations || answer
    );
    saveTurn(value, answer, sources, Boolean(data.success), data.rag || null, data.answer_with_citations || answer);
    setStatus(data.success ? "Готов" : "Недостаточно данных");
  } catch (error) {
    removeSearchProgress();
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

clearHistoryButton?.addEventListener("click", clearChat);

updateCounter();
renderHistory();
restoreHistory();
checkHealth();
question.focus();
