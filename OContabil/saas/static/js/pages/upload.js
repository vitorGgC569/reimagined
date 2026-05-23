/* ============================================================
   ContaDoc AI — Upload Page
   ============================================================ */

function renderUpload() {
    const queue = MockData.processingQueue;

    return `
    <!-- Upload Zone -->
    <div class="upload-zone" id="upload-zone"
         ondragover="handleDragOver(event)"
         ondragleave="handleDragLeave(event)"
         ondrop="handleDrop(event)"
         onclick="document.getElementById('file-input').click()">
      <div class="upload-icon">📤</div>
      <div class="upload-title">Arraste e solte documentos aqui</div>
      <div class="upload-subtitle">ou clique para selecionar · PDF, JPG, PNG, ZIP · Até 50 MB por arquivo</div>
      <input type="file" id="file-input" multiple accept=".pdf,.jpg,.jpeg,.png,.zip" style="display:none" onchange="handleFileSelect(event)" />
      <div style="margin-top:var(--space-5);display:flex;gap:var(--space-3);justify-content:center;position:relative">
        <button class="btn btn-primary btn-lg" onclick="event.stopPropagation(); document.getElementById('file-input').click()">
          📂 Selecionar Arquivos
        </button>
        <button class="btn btn-secondary btn-lg" onclick="event.stopPropagation(); showBatchUpload()">
          📦 Upload em Lote
        </button>
      </div>
    </div>

    <!-- Schema Selection -->
    <div class="card" style="margin-top:var(--space-5)">
      <div class="card-header">
        <div>
          <div class="card-title">⚙️ Schema de Extração GLiNER 2</div>
          <div class="card-subtitle">Selecione o tipo de documento ou use detecção automática</div>
        </div>
        <span class="badge badge-info">Zero-Shot</span>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:var(--space-3)">
        <div class="schema-option selected" data-schema="auto" onclick="selectSchema('auto', this)">
          <div style="font-size:24px;margin-bottom:var(--space-2)">🤖</div>
          <div style="font-weight:600;font-size:var(--text-sm)">Detecção Automática</div>
          <div style="font-size:var(--text-xs);color:var(--text-tertiary)">IA identifica o tipo</div>
        </div>
        ${Object.entries(MockData.extractionSchemas).map(([key, schema]) => `
          <div class="schema-option" data-schema="${key}" onclick="selectSchema('${key}', this)">
            <div style="font-size:24px;margin-bottom:var(--space-2)">${key === 'nota_fiscal' ? '📃' :
            key === 'boleto' ? '🏦' :
                key === 'comprovante_pix' ? '💸' : '📋'
        }</div>
            <div style="font-weight:600;font-size:var(--text-sm)">${schema.label}</div>
            <div style="font-size:var(--text-xs);color:var(--text-tertiary)">${schema.fields.length} campos</div>
          </div>
        `).join('')}
      </div>
    </div>

    <!-- Processing Queue -->
    <div class="card" style="margin-top:var(--space-5)">
      <div class="card-header">
        <div>
          <div class="card-title">📋 Fila de Processamento</div>
          <div class="card-subtitle">${queue.filter(q => q.status !== 'completed').length} itens em processamento</div>
        </div>
        <div style="display:flex;gap:var(--space-2)">
          <button class="btn btn-sm btn-ghost">🔄 Atualizar</button>
          <button class="btn btn-sm btn-secondary">⏸️ Pausar Fila</button>
        </div>
      </div>
      <div class="queue-list" id="queue-list">
        ${queue.map(q => renderQueueItem(q)).join('')}
      </div>
    </div>

    <style>
      .schema-option {
        padding: var(--space-4);
        background: var(--surface-card);
        border: 2px solid var(--border-default);
        border-radius: var(--border-radius-md);
        text-align: center;
        cursor: pointer;
        transition: all var(--transition-fast);
      }
      .schema-option:hover {
        border-color: var(--border-hover);
        background: var(--surface-card-hover);
      }
      .schema-option.selected {
        border-color: var(--color-accent);
        background: var(--color-accent-glow);
      }
    </style>
  `;
}

function renderQueueItem(q) {
    const statusConfig = {
        processing: { badge: 'badge-info', label: 'Processando', icon: '⚙️' },
        queued: { badge: 'badge-neutral', label: 'Na Fila', icon: '⏳' },
        completed: { badge: 'badge-success', label: 'Concluído', icon: '✅' },
        error: { badge: 'badge-error', label: 'Erro', icon: '❌' },
    };
    const cfg = statusConfig[q.status] || statusConfig.queued;

    return `
    <div class="queue-item">
      <div class="file-icon">${cfg.icon}</div>
      <div class="file-info">
        <div class="file-name">${q.filename}</div>
        <div class="file-meta">
          <span>${q.size}</span>
          <span>·</span>
          <span>${q.filesCount} arquivo${q.filesCount > 1 ? 's' : ''}</span>
        </div>
      </div>
      <span class="badge ${cfg.badge}">${cfg.label}</span>
      ${q.status === 'processing' ? `
        <div class="progress-bar" style="width:150px">
          <div class="progress-fill" style="width:${q.progress}%"></div>
        </div>
        <span style="font-size:var(--text-xs);font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-secondary);min-width:35px">${q.progress}%</span>
      ` : ''}
      ${q.status === 'completed' ? `
        <button class="btn btn-sm btn-ghost" onclick="navigateTo('validation')">Revisar →</button>
      ` : ''}
    </div>
  `;
}

// ── Upload Handlers ────────────────────────────────────────
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('upload-zone').classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        simulateUpload(files);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        simulateUpload(files);
    }
}

function simulateUpload(files) {
    const count = files.length;
    const names = Array.from(files).map(f => f.name).join(', ');

    // Add to queue
    const newItem = {
        id: 'Q-' + Date.now(),
        filename: count > 1 ? `${count} arquivos selecionados` : files[0].name,
        size: formatFileSize(Array.from(files).reduce((sum, f) => sum + f.size, 0)),
        filesCount: count,
        status: 'processing',
        progress: 0,
    };

    const queueList = document.getElementById('queue-list');
    if (queueList) {
        queueList.insertAdjacentHTML('afterbegin', renderQueueItem(newItem));
    }

    // Update badge
    const badge = document.getElementById('upload-badge');
    if (badge) {
        const current = parseInt(badge.textContent) || 0;
        badge.textContent = current + count;
        badge.style.display = 'block';
    }

    showToast(`${count} arquivo${count > 1 ? 's' : ''} enviado${count > 1 ? 's' : ''} para processamento`, 'success');

    // Simulate progress
    simulateProgress(newItem.id);
}

function simulateProgress(itemId) {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.floor(Math.random() * 15) + 5;
        if (progress >= 100) {
            progress = 100;
            clearInterval(interval);
            showToast('Documento processado com sucesso!', 'success');
        }
    }, 800);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function selectSchema(schemaKey, element) {
    document.querySelectorAll('.schema-option').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');
    showToast(`Schema "${schemaKey === 'auto' ? 'Detecção Automática' : MockData.extractionSchemas[schemaKey]?.label || schemaKey}" selecionado`, 'success');
}

function showBatchUpload() {
    showModal('Upload em Lote', `
    <div class="form-group">
      <label class="form-label">Cliente Destino</label>
      <select class="form-input">
        <option value="">— Selecionar cliente —</option>
        ${MockData.clients.map(c => `<option value="${c.id}">${c.name}</option>`).join('')}
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Tipo de Documento</label>
      <select class="form-input">
        <option value="auto">Detecção Automática</option>
        ${Object.entries(MockData.extractionSchemas).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('')}
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Período de Referência</label>
      <input type="month" class="form-input" value="2026-03" />
    </div>
    <div class="upload-zone" style="padding:var(--space-8)" onclick="document.getElementById('batch-file-input').click()">
      <div style="font-size:32px">📦</div>
      <div style="font-weight:600;margin-top:var(--space-2)">Arraste o arquivo ZIP ou pasta</div>
      <div style="font-size:var(--text-xs);color:var(--text-tertiary)">ZIP com múltiplos PDFs e imagens</div>
      <input type="file" id="batch-file-input" accept=".zip" style="display:none" />
    </div>
  `, `
    <button class="btn btn-secondary" onclick="closeModal()">Cancelar</button>
    <button class="btn btn-primary" onclick="closeModal(); showToast('Lote enviado para processamento!', 'success')">🚀 Processar Lote</button>
  `);
}
