const form = document.querySelector('#form');
const input = document.querySelector('#input');
const chat = document.querySelector('#chat');
const button = form.querySelector('button');

function addMessage(role, text) {
  const article = document.createElement('article');
  article.className = `message ${role}`;
  const small = document.createElement('small');
  small.textContent = role === 'user' ? 'Você' : 'VERD';
  const p = document.createElement('p');
  p.textContent = text;
  article.append(small, p);
  chat.appendChild(article);
  chat.scrollTop = chat.scrollHeight;
}

fetch('/api/info').then(r => r.json()).then(info => {
  document.querySelector('#modelInfo').textContent = `${Number(info.parameters || 0).toLocaleString('pt-BR')} parâmetros • BRZ${info.version}`;
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  addMessage('user', message);
  input.value = '';
  button.disabled = true;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message,
        temperature: Number(document.querySelector('#temperature').value),
        max_tokens: Number(document.querySelector('#maxTokens').value),
      }),
    });
    const data = await response.json();
    addMessage('assistant', data.answer ?? `Erro: ${data.error}`);
  } catch (error) {
    addMessage('assistant', `Erro de conexão: ${error}`);
  } finally {
    button.disabled = false;
    input.focus();
  }
});
