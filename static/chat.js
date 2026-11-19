const form = document.querySelector('#question-form');
const input = document.querySelector('#question');
const send = document.querySelector('#send');
const clearChat = document.querySelector('#clear-chat');
const conversation = document.querySelector('#conversation');
const history = document.querySelector('#history');
const suggestions = document.querySelector('#suggestions');
const typing = document.querySelector('#typing');
const status = document.querySelector('#status');
const starters = document.querySelectorAll('[data-question]');
const avatar = typing.querySelector('.avatar');
const topicsToggle = document.querySelector('#topics-toggle');

let pendingRequest = null;
let retryButton = null;

function updateControls() {
  const hasDraft = Boolean(input.value.trim());
  send.disabled = Boolean(pendingRequest) || !hasDraft;
  clearChat.disabled = !history.childElementCount && !hasDraft;
  for (const button of starters) button.disabled = Boolean(pendingRequest);

  input.style.height = 'auto';
  input.style.height = input.scrollHeight + 'px';
}

function setTyping(isTyping) {
  typing.hidden = !isTyping;
  status.textContent = isTyping ? 'TinyRAG Assistant is typing.' : '';
  updateControls();
}

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function isNearLatest() {
  return conversation.scrollHeight - conversation.clientHeight - conversation.scrollTop < 48;
}

function setTopicsExpanded(expanded) {
  const follow = isNearLatest();
  topicsToggle.setAttribute('aria-expanded', String(expanded));
  if (follow) scrollToLatest();
}

function addMessage(text, kind) {
  const turn = document.createElement('div');
  turn.className = `turn ${kind}`;
  const message = document.createElement('div');
  message.className = 'message';
  const label = document.createElement('p');
  label.className = 'speaker';
  label.textContent = kind === 'user' ? 'You' : 'TinyRAG Assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (kind === 'assistant') {
    bubble.classList.add('markdown');
    // Only the server's rendered Markdown goes through this branch.
    bubble.innerHTML = text;
  } else {
    bubble.textContent = text;
  }

  if (kind !== 'user') turn.append(avatar.cloneNode(true));
  message.append(label, bubble);
  turn.append(message);
  history.append(turn);
  return { turn, bubble };
}

function showError(question, text) {
  const { turn, bubble } = addMessage(text, 'error');
  retryButton = document.createElement('button');
  retryButton.type = 'button';
  retryButton.className = 'retry';
  retryButton.textContent = 'Try again';
  retryButton.addEventListener('click', () => {
    turn.remove();
    conversation.focus({ preventScroll: true });
    ask(question, { retry: true });
  });
  bubble.append(retryButton);
}

async function ask(question, { retry = false } = {}) {
  if (!question || pendingRequest) return;

  const controller = new AbortController();
  pendingRequest = controller;
  retryButton?.remove();
  retryButton = null;
  suggestions.hidden = true;

  if (!retry) addMessage(question, 'user');
  setTyping(true);
  scrollToLatest();

  let answerHtml;
  let error;
  try {
    const response = await fetch(form.action, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    const result = await response.json();
    if (response.ok && typeof result.answer_html === 'string') {
      answerHtml = result.answer_html.trim();
    } else if (!response.ok && typeof result.error === 'string') {
      error = result.error;
    }
  } catch {
    error = 'The request failed. Please try again.';
  }

  if (pendingRequest !== controller) return;

  const follow = isNearLatest();
  pendingRequest = null;
  setTyping(false);
  if (answerHtml) {
    addMessage(answerHtml, 'assistant');
  } else {
    showError(question, error || 'The assistant couldn’t reply. Please try again.');
  }
  // Keep the start of a long reply in view. Short replies scroll to the bottom.
  if (follow) conversation.scrollTop = history.lastElementChild.offsetTop - 16;
}

clearChat.addEventListener('click', () => {
  pendingRequest?.abort();
  pendingRequest = null;
  retryButton = null;
  history.replaceChildren();
  suggestions.hidden = false;
  input.value = '';
  conversation.focus({ preventScroll: true });
  setTopicsExpanded(false);
  setTyping(false);
  conversation.scrollTop = 0;
});

for (const button of starters) {
  button.addEventListener('click', () => {
    conversation.focus({ preventScroll: true });
    setTopicsExpanded(false);
    ask(button.dataset.question);
  });
}

topicsToggle.addEventListener('click', () => {
  setTopicsExpanded(topicsToggle.getAttribute('aria-expanded') === 'false');
});

input.addEventListener('input', updateControls);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || pendingRequest) return;
  input.value = '';
  input.focus({ preventScroll: true });
  ask(question);
});

updateControls();
