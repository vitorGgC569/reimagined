/* ============================================================
   ContaDoc AI — Validation Station (Core Product Screen)
   Split-screen document viewer + extraction form
   ============================================================ */

let currentDocIndex = 0;
let validationDocs = [];

function renderValidation() {
    validationDocs = MockData.validationQueue;

    if (validationDocs.length === 0) {
        return `
      <div class="empty-state">
        <div class="empty-icon">✅</div>
        <div class="empty-title">Nenhum documento pendente</div>
        <div class="empty-text">Todos os documentos foram validados. Envie mais documentos na página de Upload.</div>
        <button class="btn btn-primary" style="margin-top:var(--space-5)" onclick="navigateTo('upload')">📤 Ir para Upload</button>
      </div>
    `;
    }

    const doc = validationDocs[currentDocIndex];

    return `
    <!-- Validation Nav Bar -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--space-4)">
      <div style="display:flex;align-items:center;gap:var(--space-3)">
        <button class="btn btn-sm btn-secondary" onclick="prevDocument()" ${currentDocIndex === 0 ? 'disabled style="opacity:0.4;pointer-events:none"' : ''}>
          ← Anterior
        </button>
        <span style="font-size:var(--text-sm);color:var(--text-secondary);font-variant-numeric:tabular-nums">
          ${currentDocIndex + 1} de ${validationDocs.length}
        </span>
        <button class="btn btn-sm btn-secondary" onclick="nextDocument()" ${currentDocIndex === validationDocs.length - 1 ? 'disabled style="opacity:0.4;pointer-events:none"' : ''}>
          Próximo →
        </button>
      </div>
      <div style="display:flex;align-items:center;gap:var(--space-3)">
        <span class="kbd">Tab</span> <span style="font-size:var(--text-xs);color:var(--text-tertiary)">navegar campos</span>
        <span class="kbd">Ctrl+Enter</span> <span style="font-size:var(--text-xs);color:var(--text-tertiary)">validar</span>
        <span class="kbd">Ctrl+→</span> <span style="font-size:var(--text-xs);color:var(--text-tertiary)">próximo</span>
      </div>
    </div>

    <!-- Document Queue Chips -->
    <div style="display:flex;gap:var(--space-2);margin-bottom:var(--space-4);overflow-x:auto;padding-bottom:var(--space-2)">
      ${validationDocs.map((d, i) => `
        <button class="btn btn-sm ${i === currentDocIndex ? 'btn-primary' : 'btn-secondary'}" 
                onclick="switchDocument(${i})"
                style="flex-shrink:0">
          ${getDocTypeIcon(d.type)} ${d.id}
        </button>
      `).join('')}
    </div>

    <!-- Split Screen -->
    <div class="split-screen">
      <!-- Left: Document Viewer -->
      <div class="split-left">
        <div class="doc-viewer">
          <div class="doc-viewer-toolbar">
            <span style="font-size:var(--text-sm);font-weight:600;color:var(--text-primary)">${doc.filename}</span>
            <span class="badge badge-info">${doc.type}</span>
            <span style="flex:1"></span>
            <button class="btn btn-sm btn-ghost" title="Zoom In">🔍+</button>
            <button class="btn btn-sm btn-ghost" title="Zoom Out">🔍−</button>
            <button class="btn btn-sm btn-ghost" title="Rotacionar">🔄</button>
          </div>
          <div class="doc-viewer-body" id="doc-viewer-body">
            ${renderDocumentContent(doc)}
          </div>
        </div>
      </div>

      <!-- Right: Extraction Form -->
      <div class="split-right">
        <div class="validation-form">
          <div class="validation-form-header">
            <div>
              <div style="font-size:var(--text-base);font-weight:600">Dados Extraídos</div>
              <div style="font-size:var(--text-xs);color:var(--text-tertiary)">GLiNER 2 · Confiança média: ${calculateAvgConfidence(doc.extracted)}%</div>
            </div>
            <div style="display:flex;align-items:center;gap:var(--space-3)">
              ${renderOverallConfidence(doc.extracted)}
            </div>
          </div>
          <div class="validation-form-body" id="validation-form-body">
            ${renderExtractionForm(doc)}
          </div>
          <div class="validation-form-footer">
            <div style="display:flex;gap:var(--space-2)">
              <button class="btn btn-sm btn-ghost" onclick="rejectDocument()">❌ Rejeitar</button>
              <button class="btn btn-sm btn-secondary" onclick="skipDocument()">⏩ Pular</button>
            </div>
            <div style="display:flex;gap:var(--space-2)">
              <button class="btn btn-sm btn-secondary" onclick="showExportOptions()">📥 Exportar JSON</button>
              <button class="btn btn-success" onclick="validateDocument()" id="btn-validate">
                ✅ Validar e Próximo <span class="kbd" style="margin-left:var(--space-2);color:inherit;background:rgba(255,255,255,0.15);border-color:rgba(255,255,255,0.2)">Ctrl+↵</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderDocumentContent(doc) {
    // Render the OCR text with highlighted spans
    let text = doc.ocrText;
    const fields = doc.extracted;
    let highlights = [];

    Object.entries(fields).forEach(([fieldName, data]) => {
        if (data.span) {
            highlights.push({
                field: fieldName,
                value: data.value,
                confidence: data.confidence,
                start: data.span[0],
                end: data.span[1],
            });
        }
    });

    // Sort highlights by position (reverse to not mess up indices)
    highlights.sort((a, b) => a.start - b.start);

    // Build highlighted text
    let result = '';
    let lastEnd = 0;

    highlights.forEach(h => {
        const confClass = h.confidence >= 0.9 ? 'high' : h.confidence >= 0.8 ? 'medium' : 'low';
        const confColor = confClass === 'high' ? 'var(--color-success)' : confClass === 'medium' ? 'var(--color-warning)' : 'var(--color-error)';

        result += escapeHtml(text.substring(lastEnd, h.start));
        result += `<span class="ocr-highlight ${confClass}" 
                     data-field="${h.field}" 
                     title="${formatFieldName(h.field)}: ${h.value} (${Math.round(h.confidence * 100)}%)"
                     onclick="focusField('${h.field}')"
                     style="background:${confColor}20;border-bottom:2px solid ${confColor};cursor:pointer;padding:1px 2px;border-radius:2px;transition:all 0.15s">`;
        result += escapeHtml(text.substring(h.start, h.end));
        result += '</span>';
        lastEnd = h.end;
    });

    result += escapeHtml(text.substring(lastEnd));

    return `
    <pre style="font-family:var(--font-mono);font-size:13px;line-height:1.7;white-space:pre-wrap;word-wrap:break-word;color:var(--text-primary);padding:var(--space-4)">${result}</pre>
    <div style="padding:var(--space-4);border-top:1px solid var(--border-default);display:flex;gap:var(--space-4);font-size:var(--text-xs);color:var(--text-tertiary)">
      <span>🟢 Alta confiança (≥90%)</span>
      <span>🟡 Média confiança (80-89%)</span>
      <span>🔴 Baixa confiança (<80%)</span>
      <span style="margin-left:auto">Clique em um trecho destacado para navegar ao campo</span>
    </div>
  `;
}

function renderExtractionForm(doc) {
    const fields = doc.extracted;

    return Object.entries(fields).map(([fieldName, data]) => {
        const confPercent = Math.round(data.confidence * 100);
        const confClass = data.confidence >= 0.9 ? 'high' : data.confidence >= 0.8 ? 'medium' : 'low';
        const inputClass = `confidence-${confClass}`;
        const confColor = confClass === 'high' ? 'var(--color-success)' : confClass === 'medium' ? 'var(--color-warning)' : 'var(--color-error)';

        return `
      <div class="form-group" id="field-group-${fieldName}" style="animation:fadeIn 0.3s ease">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--space-2)">
          <label class="form-label" style="margin-bottom:0">${formatFieldName(fieldName)}</label>
          <div class="confidence-indicator" style="color:${confColor}">
            <div class="confidence-meter" style="width:80px">
              <div class="confidence-bar">
                <div class="confidence-fill ${confClass}" style="width:${confPercent}%"></div>
              </div>
              <span class="confidence-value" style="color:${confColor}">${confPercent}%</span>
            </div>
          </div>
        </div>
        <input type="text" 
               class="form-input ${inputClass}" 
               id="field-${fieldName}" 
               value="${data.value}" 
               data-field="${fieldName}"
               data-original="${data.value}"
               onfocus="highlightInDoc('${fieldName}')"
               onchange="markAsEdited(this)"
               tabindex="0" />
        ${confPercent < 80 ? `
          <div class="form-hint" style="color:var(--color-warning)">
            ⚠️ Confiança baixa — revise manualmente este campo
          </div>
        ` : ''}
      </div>
    `;
    }).join('');
}

function renderOverallConfidence(extracted) {
    const avg = calculateAvgConfidence(extracted);
    const confClass = avg >= 90 ? 'high' : avg >= 80 ? 'medium' : 'low';
    const confColor = confClass === 'high' ? 'var(--color-success)' : confClass === 'medium' ? 'var(--color-warning)' : 'var(--color-error)';

    return `
    <div style="display:flex;align-items:center;gap:var(--space-2)">
      <svg width="32" height="32" viewBox="0 0 36 36" style="transform:rotate(-90deg)">
        <circle r="15" cx="18" cy="18" fill="none" stroke="var(--border-default)" stroke-width="3"></circle>
        <circle r="15" cx="18" cy="18" fill="none" stroke="${confColor}" stroke-width="3"
                stroke-dasharray="${avg * 0.942} 100" stroke-linecap="round"></circle>
      </svg>
      <span style="font-size:var(--text-sm);font-weight:700;color:${confColor}">${avg}%</span>
    </div>
  `;
}

// ── Helper Functions ─────────────────────────────────────────
function calculateAvgConfidence(extracted) {
    const values = Object.values(extracted).map(d => d.confidence);
    return Math.round((values.reduce((a, b) => a + b, 0) / values.length) * 100);
}

function formatFieldName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function getDocTypeIcon(type) {
    const icons = {
        'Nota Fiscal': '📃',
        'Boleto': '🏦',
        'Comprovante PIX': '💸',
        'Guia DARF': '📋',
        'Extrato Bancário': '🏧',
    };
    return icons[type] || '📄';
}

// ── Navigation ───────────────────────────────────────────────
function switchDocument(index) {
    currentDocIndex = index;
    document.getElementById('content-body').innerHTML = renderValidation();
}

function prevDocument() {
    if (currentDocIndex > 0) {
        currentDocIndex--;
        document.getElementById('content-body').innerHTML = renderValidation();
    }
}

function nextDocument() {
    if (currentDocIndex < validationDocs.length - 1) {
        currentDocIndex++;
        document.getElementById('content-body').innerHTML = renderValidation();
    }
}

// ── Actions ──────────────────────────────────────────────────
function validateDocument() {
    const doc = validationDocs[currentDocIndex];
    showToast(`Documento ${doc.id} validado com sucesso!`, 'success');

    // Remove from queue
    validationDocs.splice(currentDocIndex, 1);

    // Update badge
    const badge = document.getElementById('validation-badge');
    if (badge) {
        badge.textContent = validationDocs.length;
        if (validationDocs.length === 0) badge.style.display = 'none';
    }

    if (currentDocIndex >= validationDocs.length) {
        currentDocIndex = Math.max(0, validationDocs.length - 1);
    }

    document.getElementById('content-body').innerHTML = renderValidation();
}

function rejectDocument() {
    const doc = validationDocs[currentDocIndex];
    showToast(`Documento ${doc.id} rejeitado — será reprocessado`, 'warning');
    nextDocument();
}

function skipDocument() {
    nextDocument();
}

function highlightInDoc(fieldName) {
    // Remove previous highlights
    document.querySelectorAll('.ocr-highlight').forEach(el => {
        el.style.outline = 'none';
        el.style.boxShadow = 'none';
    });

    // Highlight the matching span in the document
    const target = document.querySelector(`.ocr-highlight[data-field="${fieldName}"]`);
    if (target) {
        target.style.outline = '2px solid var(--color-accent)';
        target.style.boxShadow = '0 0 8px var(--color-accent-glow)';
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function focusField(fieldName) {
    const input = document.getElementById(`field-${fieldName}`);
    if (input) {
        input.focus();
        input.select();

        // Scroll form to field
        const group = document.getElementById(`field-group-${fieldName}`);
        if (group) {
            group.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
}

function markAsEdited(input) {
    const original = input.dataset.original;
    if (input.value !== original) {
        input.style.borderColor = 'var(--color-accent)';
        input.style.boxShadow = '0 0 0 2px var(--color-accent-glow)';
    }
}

function showExportOptions() {
    const doc = validationDocs[currentDocIndex];
    const jsonData = {};
    Object.entries(doc.extracted).forEach(([key, data]) => {
        jsonData[key] = data.value;
    });

    showModal('Exportar Dados Extraídos', `
    <div class="tabs">
      <div class="tab active" onclick="switchExportTab('json', this)">JSON</div>
      <div class="tab" onclick="switchExportTab('csv', this)">CSV</div>
      <div class="tab" onclick="switchExportTab('api', this)">API</div>
    </div>
    <div id="export-content">
      <pre style="background:var(--surface-input);padding:var(--space-4);border-radius:var(--border-radius-sm);font-family:var(--font-mono);font-size:var(--text-sm);overflow-x:auto;color:var(--color-accent-light)">${JSON.stringify(jsonData, null, 2)}</pre>
    </div>
  `, `
    <button class="btn btn-secondary" onclick="closeModal()">Fechar</button>
    <button class="btn btn-primary" onclick="copyToClipboard(); showToast('JSON copiado!', 'success')">📋 Copiar</button>
  `);
}

function switchExportTab(tab, element) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    element.classList.add('active');

    const doc = validationDocs[currentDocIndex];
    const content = document.getElementById('export-content');

    if (tab === 'json') {
        const jsonData = {};
        Object.entries(doc.extracted).forEach(([key, data]) => {
            jsonData[key] = data.value;
        });
        content.innerHTML = `<pre style="background:var(--surface-input);padding:var(--space-4);border-radius:var(--border-radius-sm);font-family:var(--font-mono);font-size:var(--text-sm);overflow-x:auto;color:var(--color-accent-light)">${JSON.stringify(jsonData, null, 2)}</pre>`;
    } else if (tab === 'csv') {
        const rows = Object.entries(doc.extracted).map(([key, data]) => `${key},"${data.value}",${Math.round(data.confidence * 100)}%`);
        content.innerHTML = `<pre style="background:var(--surface-input);padding:var(--space-4);border-radius:var(--border-radius-sm);font-family:var(--font-mono);font-size:var(--text-sm);overflow-x:auto;color:var(--text-primary)">campo,valor,confianca\n${rows.join('\n')}</pre>`;
    } else {
        content.innerHTML = `
      <div class="form-group">
        <label class="form-label">URL do Endpoint (ERP)</label>
        <input type="text" class="form-input" value="https://erp.contabilidade.com.br/api/v1/documents" />
      </div>
      <div class="form-group">
        <label class="form-label">API Key</label>
        <input type="password" class="form-input" value="sk-contadoc-12345" />
      </div>
      <div style="font-size:var(--text-xs);color:var(--text-tertiary)">Os dados serão enviados via POST com autenticação Bearer Token.</div>
    `;
    }
}

function copyToClipboard() {
    const doc = validationDocs[currentDocIndex];
    const jsonData = {};
    Object.entries(doc.extracted).forEach(([key, data]) => {
        jsonData[key] = data.value;
    });
    navigator.clipboard?.writeText(JSON.stringify(jsonData, null, 2));
}
