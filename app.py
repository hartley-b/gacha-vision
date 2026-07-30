"""End-user upload-and-review web app: screenshots in -> character IDs out.

The end goal the whole pipeline builds toward (README "Planned: end-user
web app", now real): drop one or more gacha-rate-list screenshots on the
page, get back the set of detected character ids as copyable JSON-ish
output (e.g. {1002, 4032}), plus a per-row review table underneath so a
wrong row can be caught and corrected instead of silently trusted.

Per detection the table shows the same evidence the audit tools show:
  - the icon crop from the uploaded screenshot,
  - the visual match (final_sorted reference icon, id, name, score),
  - the text signal (crop of the on-screen name, the raw OCR read, and the
    reference icon/name/similarity of the id the lookup resolved to),
  - the final pick (editable id + live reference icon), and an include
    checkbox — edits and exclusions update the output set immediately.

Uploads are held in memory per batch (nothing written to disk); only the
most recent batches are kept.

Usage: python app.py, then open http://127.0.0.1:5063
"""

import sys
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import identification as ident  # noqa: E402
import matching as mt  # noqa: E402
import name_ocr  # noqa: E402

ALL_DIR = ROOT / "final_sorted" / "ALL"
_MAX_BATCHES = 8  # in-memory batches kept before the oldest is dropped

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

INDEX = mt.build_reference_index()
NAME_INDEX = name_ocr.load_name_index()

# batch_id -> {"files": [fname, ...], "images": {fname: np img},
#              "results": {fname: [IdentifiedIcon, ...]}}
_BATCHES: "OrderedDict[str, dict]" = OrderedDict()


def _names(id_: str) -> tuple[str, str]:
    return (
        NAME_INDEX.id_to_name.get(id_, "?"),
        NAME_INDEX.id_to_eng_name.get(id_, "?"),
    )


def _row_payload(fname: str, i: int, r: "ident.IdentifiedIcon") -> dict:
    jpn, eng = _names(r.id)
    vis_jpn, _ = _names(r.visual_id) if r.visual_id else ("?", "?")
    text_id = r.name_match_ids[0] if r.name_match_ids else None
    text_jpn = r.name_match_name
    return {
        "file": fname,
        "row_index": i,
        "id": r.id,
        "rank": r.rank,
        "rank_was_guessed": r.rank_was_guessed,
        "score": r.score,
        "resolved_by": r.resolved_by,
        "jpn_name": jpn,
        "eng_name": eng,
        "visual_id": r.visual_id,
        "visual_score": r.visual_score,
        "visual_jpn_name": vis_jpn if r.visual_id else None,
        "ocr_text": r.name_ocr_text,
        "text_id": text_id,
        "text_jpn_name": text_jpn,
        "text_sim": r.name_match_score,
        "name_agrees": r.name_agrees,
    }


@app.route("/")
def index():
    return render_template_string(_TEMPLATE)


@app.route("/api/identify", methods=["POST"])
def api_identify():
    files = request.files.getlist("screenshots")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    batch_id = uuid.uuid4().hex[:12]
    batch = {"files": [], "images": {}, "results": {}}
    rows = []
    for idx, f in enumerate(files):
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue  # not an image — skip rather than fail the whole batch
        # key by index + original name so duplicate filenames can't collide
        fname = f"{idx}_{Path(f.filename or 'upload.png').name}"
        results = ident.identify_screenshot(img, INDEX, NAME_INDEX)
        batch["files"].append(fname)
        batch["images"][fname] = img
        batch["results"][fname] = results
        rows.extend(_row_payload(fname, i, r) for i, r in enumerate(results))

    if not batch["files"]:
        return jsonify({"error": "none of the uploads could be read as images"}), 400

    _BATCHES[batch_id] = batch
    while len(_BATCHES) > _MAX_BATCHES:
        _BATCHES.popitem(last=False)
    return jsonify({"batch_id": batch_id, "rows": rows})


def _batch_row(batch_id: str, fname: str, row_index: int):
    batch = _BATCHES.get(batch_id)
    if batch is None:
        return None, None
    img = batch["images"].get(fname)
    results = batch["results"].get(fname)
    if img is None or results is None or row_index >= len(results):
        return None, None
    return img, results[row_index]


@app.route("/api/batch/<batch_id>/crop/<fname>/<int:row_index>.png")
def api_crop(batch_id, fname, row_index):
    img, r = _batch_row(batch_id, fname, row_index)
    if img is None:
        return "", 404
    b = r.box
    crop = img[b.y : b.y + b.h, b.x : b.x + b.w]
    _, buf = cv2.imencode(".png", crop)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/batch/<batch_id>/namecrop/<fname>/<int:row_index>.png")
def api_namecrop(batch_id, fname, row_index):
    img, r = _batch_row(batch_id, fname, row_index)
    if img is None:
        return "", 404
    region = name_ocr.find_name_region(img, r.box)
    if region is None:
        return "", 404
    x0, y0, x1, y1 = region
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return "", 404
    _, buf = cv2.imencode(".png", crop)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/icon/<id_>.png")
def api_icon(id_):
    # id_ comes from user-editable input — resolve strictly inside ALL_DIR
    path = (ALL_DIR / f"{Path(id_).name}.png").resolve()
    if path.parent != ALL_DIR.resolve():
        return "", 404
    img = cv2.imread(str(path))
    if img is None:
        return "", 404
    _, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/name/<id_>")
def api_name(id_):
    """Names for a (possibly hand-edited) id, so the table can relabel."""
    jpn, eng = _names(id_)
    return jsonify({"jpn_name": jpn, "eng_name": eng, "known": id_ in NAME_INDEX.id_to_name})


_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Puni Icons — screenshot to IDs</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; padding: 24px; max-width: 1200px; margin: 0 auto; }
  h2 { margin-top: 0; }
  #drop { display: block; border: 2px dashed #3a5fc4; border-radius: 12px; padding: 36px; text-align: center; color: #aab; cursor: pointer; transition: background .15s; }
  #drop.drag { background: #24243d; }
  #drop input { display: none; }
  #status { margin: 14px 0; color: #ff9800; min-height: 1.2em; }
  #output-wrap { display: none; margin: 18px 0; }
  #output { font-family: ui-monospace, monospace; font-size: 16px; background: #0f0f1e; border: 1px solid #2a2a44; border-radius: 8px; padding: 14px; word-break: break-all; }
  button { background: #3a5fc4; color: #fff; border: 0; border-radius: 6px; padding: 8px 16px; font-size: 14px; cursor: pointer; margin-top: 8px; }
  button:hover { background: #4a6fd4; }
  .muted { color: #888; font-size: 12px; }
  table { border-collapse: collapse; width: 100%; margin-top: 18px; }
  th { text-align: left; color: #999; font-size: 11px; text-transform: uppercase; padding: 6px 10px; position: sticky; top: 0; background: #1a1a2e; z-index: 1; }
  td { padding: 8px 10px; border-bottom: 1px solid #2a2a44; vertical-align: middle; }
  tr.excluded { opacity: 0.35; }
  tr.disagree { background: #3a2a1f; }
  td.filecell { color: #789; font-size: 11px; max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  img.icon { width: 56px; height: 56px; object-fit: contain; background: #0f0f1e; border-radius: 4px; }
  img.namecrop { max-width: 240px; max-height: 52px; object-fit: contain; background: #f4f8fb; border-radius: 4px; display: block; }
  input.id-input { width: 70px; padding: 6px; font-size: 14px; background: #24243d; border: 1px solid #444; color: #eee; border-radius: 4px; }
  input.id-input.unknown { border-color: #ff6b6b; }
  .signal { display: flex; align-items: center; gap: 8px; }
  .signal .detail { font-size: 11px; color: #ccc; }
  .signal .detail .id { color: #ffa500; font-weight: bold; }
  .signal .detail .score { color: #888; }
  .ocr-raw { font-size: 11px; color: #888; margin-top: 3px; }
  .tag { font-size: 10px; padding: 1px 5px; border-radius: 3px; background: #3a5fc4; color: #fff; margin-left: 4px; }
  .tag.warn { background: #b3542d; }
  .final { display: flex; align-items: center; gap: 8px; }
  .final .names { font-size: 11px; color: #ccc; max-width: 150px; }
</style>
</head>
<body>
  <h2>Puni Icons — screenshot &rarr; character IDs</h2>

  <label id="drop">
    <input type="file" id="file-input" multiple accept="image/*">
    <div><b>Drop screenshots here</b> or click to choose (multiple allowed)</div>
    <div class="muted">Each screenshot is scanned for icons, ranks and names — takes a few seconds per image.</div>
  </label>
  <div id="status"></div>

  <div id="output-wrap">
    <h3 style="margin-bottom:6px">Detected puni</h3>
    <div id="output"></div>
    <button id="copy-btn">Copy</button>
    <span class="muted" id="count"></span>
  </div>

  <table id="table" style="display:none">
    <thead><tr>
      <th></th><th>file</th><th>crop</th><th>visual match</th><th>text match</th><th>final pick</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>

<script>
let batchId = null;
let rows = [];  // each: server payload + {included: bool, finalId: str}

const drop = document.getElementById('drop');
const fileInput = document.getElementById('file-input');
const statusEl = document.getElementById('status');

drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag'); });
drop.addEventListener('dragleave', () => drop.classList.remove('drag'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('drag');
  if (e.dataTransfer.files.length) upload(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) upload(fileInput.files); });

async function upload(files) {
  const form = new FormData();
  for (const f of files) form.append('screenshots', f);
  statusEl.textContent = `Identifying ${files.length} screenshot(s)…`;
  try {
    const res = await fetch('/api/identify', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) { statusEl.textContent = data.error || 'upload failed'; return; }
    batchId = data.batch_id;
    rows = data.rows.map(r => ({ ...r, included: true, finalId: r.id }));
    statusEl.textContent = '';
    renderTable();
    renderOutput();
  } catch (err) {
    statusEl.textContent = 'error: ' + err;
  }
}

function outputText() {
  const ids = [...new Set(rows.filter(r => r.included).map(r => r.finalId))]
    .sort((a, b) => Number(a) - Number(b));
  return '{' + ids.join(', ') + '}';
}

function renderOutput() {
  const ids = new Set(rows.filter(r => r.included).map(r => r.finalId));
  document.getElementById('output-wrap').style.display = 'block';
  document.getElementById('output').textContent = outputText();
  document.getElementById('count').textContent =
    ` ${ids.size} unique / ${rows.filter(r => r.included).length} rows included`;
}

document.getElementById('copy-btn').addEventListener('click', async () => {
  await navigator.clipboard.writeText(outputText());
  const btn = document.getElementById('copy-btn');
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 1200);
});

function signalCell(id, detailHtml) {
  if (!id) return `<span class="muted">&mdash;</span>`;
  return `<div class="signal">
    <img class="icon" src="/api/icon/${id}.png">
    <div class="detail"><span class="id">${id}</span> ${detailHtml}</div>
  </div>`;
}

function renderTable() {
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  document.getElementById('table').style.display = rows.length ? 'table' : 'none';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    if (!r.included) tr.classList.add('excluded');
    if (r.name_agrees === false) tr.classList.add('disagree');

    const visualDetail =
      `${r.visual_jpn_name || ''}<br><span class="score">score ${r.visual_score != null ? r.visual_score.toFixed(3) : '?'}</span>`;
    const textDetail =
      `${r.text_jpn_name || ''}<br><span class="score">sim ${r.text_sim != null ? r.text_sim.toFixed(2) : '?'}</span>`;
    const nameCrop = `<img class="namecrop" src="/api/batch/${batchId}/namecrop/${encodeURIComponent(r.file)}/${r.row_index}.png"
      onerror="this.replaceWith(document.createTextNode('—'))">`;
    const tags =
      (r.resolved_by === 'text-override' ? `<span class="tag">text override</span>` : '') +
      (r.resolved_by === 'text-tiebreak' ? `<span class="tag">text tiebreak</span>` : '') +
      (r.name_agrees === false ? `<span class="tag warn">signals disagree</span>` : '') +
      (r.rank_was_guessed ? `<span class="tag warn">rank guessed</span>` : '');

    tr.innerHTML = `
      <td><input type="checkbox" ${r.included ? 'checked' : ''} data-i="${i}" class="inc"></td>
      <td class="filecell" title="${r.file}">${r.file.replace(/^\\d+_/, '')}</td>
      <td><img class="icon" src="/api/batch/${batchId}/crop/${encodeURIComponent(r.file)}/${r.row_index}.png"></td>
      <td>${signalCell(r.visual_id, visualDetail)}</td>
      <td>${signalCell(r.text_id, textDetail)}
          <div class="ocr-raw">read: ${r.ocr_text || '(nothing)'}</div>${nameCrop}</td>
      <td><div class="final">
        <img class="icon final-icon" src="/api/icon/${r.finalId}.png">
        <div>
          <input class="id-input" data-i="${i}" value="${r.finalId}">
          <div class="names">${r.rank}${r.rank_was_guessed ? '?' : ''} &middot; ${r.jpn_name}<br>${r.eng_name}</div>
        </div>
      </div>${tags}</td>
    `;
    tbody.appendChild(tr);
  });

  tbody.querySelectorAll('input.inc').forEach(cb => cb.addEventListener('change', () => {
    rows[cb.dataset.i].included = cb.checked;
    cb.closest('tr').classList.toggle('excluded', !cb.checked);
    renderOutput();
  }));
  tbody.querySelectorAll('input.id-input').forEach(inp => inp.addEventListener('change', async () => {
    const r = rows[inp.dataset.i];
    r.finalId = inp.value.trim();
    const info = await (await fetch(`/api/name/${encodeURIComponent(r.finalId)}`)).json();
    inp.classList.toggle('unknown', !info.known);
    const tr = inp.closest('tr');
    tr.querySelector('.final-icon').src = `/api/icon/${r.finalId}.png`;
    tr.querySelector('.names').innerHTML =
      `${r.rank}${r.rank_was_guessed ? '?' : ''} &middot; ${info.jpn_name}<br>${info.eng_name}`;
    renderOutput();
  }));
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("open http://127.0.0.1:5063")
    app.run(debug=False, port=5063)
