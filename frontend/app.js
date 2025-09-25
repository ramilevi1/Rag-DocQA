let currentSessionId = null;

const sessionIdInput = document.getElementById('sessionId');
const newSessionBtn = document.getElementById('newSessionBtn');
const resetSessionBtn = document.getElementById('resetSessionBtn');

const uploadForm = document.getElementById('uploadForm');
const uploadResult = document.getElementById('uploadResult');
const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('question');
const systemPrompt = document.getElementById('systemPrompt');
const answerDiv = document.getElementById('answer');
const sourcesDiv = document.getElementById('sources');

async function createNewSession() {
  const res = await fetch('/session/new', { method: 'POST' });
  const json = await res.json();
  currentSessionId = json.session_id;
  sessionIdInput.value = currentSessionId;
  // clear UI state
  uploadResult.textContent = '';
  answerDiv.textContent = '';
  sourcesDiv.innerHTML = '';
  systemPrompt.value = '';
  questionInput.value = '';
}

newSessionBtn.addEventListener('click', async () => {
  await createNewSession();
});

resetSessionBtn.addEventListener('click', async () => {
  if (!currentSessionId) {
    alert('No session to reset.');
    return;
  }
  const form = new FormData();
  form.append('session_id', currentSessionId);
  const res = await fetch('/session/reset', { method: 'POST', body: form });
  const json = await res.json();
  uploadResult.textContent = JSON.stringify(json, null, 2);
  // clear UI
  answerDiv.textContent = '';
  sourcesDiv.innerHTML = '';
});

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!currentSessionId) {
    await createNewSession();
  }
  const files = document.getElementById('files').files;
  if (!files || files.length === 0) {
    uploadResult.textContent = 'Please select one or more files.';
    return;
  }

  // Show selected file names and sizes
  const names = Array.from(files).map(f => `${f.name} (${f.size} bytes)`).join('\n');
  uploadResult.textContent = `Selected:\n${names}`;

  const empty = Array.from(files).filter(f => f.size === 0);
  if (empty.length) {
    uploadResult.textContent += `\n\nAborted: these files are empty:\n${empty.map(f => f.name).join('\n')}`;
    return;
  }

  const formData = new FormData();
  for (const f of files) formData.append('files', f);
  formData.append('session_id', currentSessionId);

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    const json = await res.json();
    uploadResult.textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    console.error(err);
    uploadResult.textContent = 'Upload failed.';
  }
});

askBtn.addEventListener('click', async () => {
  const question = questionInput.value.trim();
  const sys = systemPrompt.value;
  if (!question) {
    answerDiv.textContent = 'Please enter a question.';
    return;
  }
  if (!currentSessionId) {
    await createNewSession();
  }
  answerDiv.textContent = 'Thinking...';
  sourcesDiv.innerHTML = '';
  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question, system_prompt: sys, top_k: 8, session_id: currentSessionId })
    });
    const json = await res.json();
    answerDiv.textContent = json.answer || '(no answer)';
    renderSources(json.sources || []);
  } catch (err) {
    answerDiv.textContent = 'Error getting answer.';
    console.error(err);
  }
});

function renderSources(sources) {
  sourcesDiv.innerHTML = '';
  if (!sources.length) return;
  for (const s of sources) {
    const el = document.createElement('div');
    el.className = 'source-card';
    el.innerHTML = `
      <div><strong>Chunk:</strong> ${s.id}</div>
      <div class="small">Doc: ${s.metadata?.filename || s.document_id} | Score: ${s.score?.toFixed(3) ?? ''} | Session: ${s.metadata?.session_id ?? ''}</div>
      <details>
        <summary>Show excerpt</summary>
        <div class="mono">${escapeHtml(s.text?.slice(0, 800) || '')}${s.text && s.text.length > 800 ? '…' : ''}</div>
      </details>
    `;
    sourcesDiv.appendChild(el);
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

// Initialize with a fresh session on first load
(async function init() {
  await createNewSession();
})();
