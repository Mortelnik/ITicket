"""Веб-интерфейс демо: страница с имитацией чата с отделом поддержки.

Идея и подход позаимствованы у проекта коллег
(ai_ticket_classifier, https://github.com/leontievayulya/ai_ticket_classifier):
FastAPI отдаёт встроенную HTML-страницу, пользователь вводит текст обращения,
система показывает предсказанный отдел как «ответ поддержки».

HTML/CSS/JS сгенерированы ИИ-ассистентом и проверены командой
(см. docs/ai_usage.md). Внешних зависимостей нет — работает офлайн.
"""

from __future__ import annotations

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ticket-router — демо маршрутизации</title>
<style>
  :root {
    --bg: #f4f4ef; --card: #ffffff; --ink: #1a1a1a; --muted: #6b6b6b;
    --accent: #20808d; --accent-dark: #14545e; --mine: #e3e9e6; --error: #b3403a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg); color: var(--ink); min-height: 100vh;
    display: flex; align-items: center; justify-content: center; padding: 24px;
  }
  .shell { width: 100%; max-width: 720px; }
  .brand {
    display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
    color: var(--accent-dark); font-weight: 700; letter-spacing: .3px;
  }
  .brand .dot { width: 12px; height: 12px; border-radius: 3px; background: var(--accent); }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 20px; }
  .card {
    background: var(--card); border-radius: 14px; padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
  }
  label { display: block; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
  input, textarea {
    width: 100%; border: 1px solid #d9d9d3; border-radius: 8px;
    padding: 10px 12px; font: inherit; background: #fbfbf8;
  }
  textarea { resize: vertical; min-height: 90px; }
  button {
    margin-top: 16px; width: 100%; padding: 12px; border: 0; border-radius: 8px;
    background: var(--accent); color: #fff; font-weight: 600; font-size: 15px;
    cursor: pointer;
  }
  button:hover { background: var(--accent-dark); }
  button:disabled { opacity: .6; cursor: wait; }
  /* чат */
  #chat { margin-top: 20px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 15px; line-height: 1.45; }
  .msg.user { align-self: flex-end; background: var(--mine); border-bottom-right-radius: 3px; white-space: pre-wrap; }
  .msg.support { align-self: flex-start; background: var(--card); border: 1px solid #e2e6e2; border-bottom-left-radius: 3px; }
  .msg.support b { color: var(--accent-dark); }
  .msg.error { align-self: center; background: #f9ecec; color: var(--error); border: 1px solid #efc7c4; font-size: 14px; }
  .hint { color: var(--muted); font-size: 13px; margin-top: 12px; }
  code { background: #eef1ee; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
</style>
</head>
<body>
<div class="shell">
  <div class="brand"><span class="dot"></span> ticket-router</div>
  <h1>Демо: автоматическая маршрутизация тикетов</h1>
  <p class="sub">Введите обращение — модель определит отдел (queue) и «переключит» чат на него.</p>

  <div class="card">
    <label for="subject">Тема обращения</label>
    <input id="subject" type="text" placeholder="Например: Login failed">
    <label for="body">Текст обращения</label>
    <textarea id="body" placeholder="Опишите проблему: Cannot log into my account, error 500 since yesterday"></textarea>
    <button id="send" onclick="sendTicket()">Отправить в поддержку</button>
    <p class="hint">Демо использует тот же эндпоинт <code>POST /predict</code>, что и боевой API. Модель: TF-IDF + LinearSVC.</p>
  </div>

  <div id="chat"></div>
</div>

<script>
async function sendTicket() {
  const subject = document.getElementById('subject').value.trim();
  const body = document.getElementById('body').value.trim();
  const chat = document.getElementById('chat');
  const btn = document.getElementById('send');

  if (!subject && !body) {
    chat.insertAdjacentHTML('beforeend',
      '<div class="msg error">Заполните тему или текст обращения.</div>');
    return;
  }
  const text = (subject ? subject + '\\n' : '') + body;

  chat.insertAdjacentHTML('beforeend',
    '<div class="msg user"></div>');
  chat.lastElementChild.textContent = text;

  btn.disabled = true; btn.textContent = 'Маршрутизация…';
  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject: subject, body: body })
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || 'Ошибка ' + resp.status);
    }
    chat.insertAdjacentHTML('beforeend',
      '<div class="msg support">Здравствуйте! Ваше обращение направлено в отдел ' +
      '<b>' + escapeHtml(data.queue) + '</b>. Специалист уже подключился к чату и поможет вам.</div>');
  } catch (err) {
    chat.insertAdjacentHTML('beforeend',
      '<div class="msg error">Не удалось получить ответ: ' + escapeHtml(String(err.message)) + '</div>');
  } finally {
    btn.disabled = false; btn.textContent = 'Отправить в поддержку';
    chat.scrollTop = chat.scrollHeight;
  }
}
function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>
"""
