'use strict';

const $ = (selector) => document.querySelector(selector);

function element(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  Object.assign(el, props);
  el.append(...children);
  return el;
}

async function problemMessage(response) {
  try {
    const problem = await response.json();
    return problem.detail || problem.title || response.statusText;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

// ---------- tabs ----------
for (const tab of document.querySelectorAll('.tab')) {
  tab.addEventListener('click', () => {
    for (const el of document.querySelectorAll('.tab, .panel')) el.classList.remove('active');
    tab.classList.add('active');
    $(`#${tab.dataset.tab}`).classList.add('active');
    if (tab.dataset.tab === 'docs') loadDocuments();
  });
}

// ---------- server-sent events over fetch (EventSource cannot POST) ----------
async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replaceAll('\r\n', '\n');
    let end;
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = 'message';
      const data = [];
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data.push(line.slice(5));
      }
      if (data.length > 0) yield { event, data: JSON.parse(data.join('\n')) };
    }
  }
}

// ---------- chat ----------
const askForm = $('#ask-form');
const questionInput = $('#question');

questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) askForm.requestSubmit();
});

askForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  $('#ask-button').disabled = true;
  $('#status').textContent = 'Retrieving…';
  $('#stats').textContent = '';
  renderSources([]);
  let answer = '';
  let sourceCount = 0;
  renderAnswer(answer, sourceCount);

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error(await problemMessage(response));
    for await (const { event, data } of readEvents(response)) {
      if (event === 'sources') {
        sourceCount = data.sources.length;
        renderSources(data.sources);
        $('#status').textContent = 'Generating…';
      } else if (event === 'token') {
        answer += data.text;
        renderAnswer(answer, sourceCount);
      } else if (event === 'done') {
        renderStats(data);
      } else if (event === 'error') {
        throw new Error(data.message);
      }
    }
    $('#status').textContent = '';
  } catch (error) {
    $('#status').textContent = `Error: ${error.message}`;
  } finally {
    $('#ask-button').disabled = false;
  }
});

// Model output is rendered as text nodes only; [n] becomes a button when source n exists.
function renderAnswer(text, sourceCount) {
  const nodes = text.split(/(\[\d+\])/).map((part) => {
    const n = Number(part.match(/^\[(\d+)\]$/)?.[1]);
    if (n >= 1 && n <= sourceCount) {
      return element('button', {
        type: 'button', className: 'cite', textContent: String(n), title: `Show source ${n}`,
        onclick: () => showSource(n),
      });
    }
    return document.createTextNode(part);
  });
  $('#answer').replaceChildren(...nodes);
}

function renderSources(sources) {
  $('#sources-heading').hidden = sources.length === 0;
  $('#sources').replaceChildren(...sources.map((s) => element('li', { id: `source-${s.n}` },
    element('div', { className: 'source-head' },
      element('strong', { textContent: s.breadcrumb ? `${s.title} › ${s.breadcrumb}` : s.title }),
      element('span', { className: 'score', textContent: scoreText(s) })),
    element('div', { className: 'source-path', textContent: s.sourcePath }),
    element('details', {},
      element('summary', { textContent: 'Chunk text' }),
      element('pre', { textContent: s.text })))));
}

// The score of each stage that returned the chunk; vector search is shown as "similarity".
function scoreText(s) {
  const entries = Object.entries(s.scores || {});
  if (entries.length === 0) return s.score == null ? '' : `score ${s.score.toFixed(3)}`;
  return entries.map(([stage, value]) => `${stage === 'vector' ? 'similarity' : stage} ${value.toFixed(3)}`).join(' · ');
}

function showSource(n) {
  const item = $(`#source-${n}`);
  if (!item) return;
  item.querySelector('details').open = true;
  item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  item.classList.remove('flash');
  void item.offsetWidth; // restart the highlight animation
  item.classList.add('flash');
}

function renderStats(done) {
  const tokens = done.promptTokens == null ? '' : ` · ${done.promptTokens} prompt + ${done.completionTokens} completion tokens`;
  $('#stats').textContent = `retrieval ${done.retrievalMillis} ms · generation ${done.generationMillis} ms${tokens}`;
}

// ---------- documents ----------
async function loadDocuments() {
  const response = await fetch('/api/documents');
  if (!response.ok) {
    $('#docs-status').textContent = `Error: ${await problemMessage(response)}`;
    return;
  }
  const docs = await response.json();
  $('#doc-count').textContent = `${docs.length} documents`;
  $('#doc-rows').replaceChildren(...docs.map((doc) => element('tr', {},
    element('td', { textContent: doc.sourcePath }),
    element('td', { textContent: doc.title }),
    element('td', { className: 'num', textContent: String(doc.chunkCount) }),
    element('td', { textContent: doc.origin.toLowerCase() }),
    element('td', { textContent: new Date(doc.ingestedAt).toLocaleString() }),
    element('td', {}, element('button', {
      type: 'button', className: 'link', textContent: 'Delete', onclick: () => deleteDocument(doc),
    })))));
}

async function deleteDocument(doc) {
  const response = await fetch(`/api/documents/${doc.id}`, { method: 'DELETE' });
  $('#docs-status').textContent = response.ok ? `Deleted ${doc.sourcePath}` : `Error: ${await problemMessage(response)}`;
  loadDocuments();
}

$('#ingest-corpus').addEventListener('click', async (e) => {
  const button = e.currentTarget;
  button.disabled = true;
  $('#docs-status').textContent = 'Ingesting corpus… (embedding every new or changed page)';
  try {
    const response = await fetch('/api/ingest/corpus', { method: 'POST' });
    if (!response.ok) throw new Error(await problemMessage(response));
    const r = await response.json();
    const failed = r.failed.length > 0 ? ` · failed (see server log): ${r.failed.join(', ')}` : '';
    $('#docs-status').textContent =
      `Added ${r.added}, updated ${r.updated}, unchanged ${r.skipped}, removed ${r.removed} · ${r.chunksWritten} chunks embedded${failed}`;
  } catch (error) {
    $('#docs-status').textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    loadDocuments();
  }
});

$('#upload-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = $('#file').files[0];
  if (!file) return;
  const body = new FormData();
  body.append('file', file);
  $('#docs-status').textContent = `Uploading ${file.name}…`;
  const response = await fetch('/api/documents', { method: 'POST', body });
  if (response.ok) {
    const outcome = await response.json();
    $('#docs-status').textContent = `${file.name}: ${outcome.status.toLowerCase()} (${outcome.document.chunkCount} chunks)`;
    e.target.reset();
  } else {
    $('#docs-status').textContent = `Error: ${await problemMessage(response)}`;
  }
  loadDocuments();
});
