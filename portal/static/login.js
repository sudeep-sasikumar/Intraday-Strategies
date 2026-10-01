document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const err = document.getElementById('err');
  err.hidden = true;
  const r = await fetch('/api/login', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({password: document.getElementById('pw').value})});
  if (r.ok) { location.href = '/'; return; }
  err.textContent = r.status === 401 ? 'Wrong password.' : 'Could not sign in. Try again.';
  err.hidden = false;
});
