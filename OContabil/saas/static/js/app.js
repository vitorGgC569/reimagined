/* ============================================================
   ContaDoc AI — Main Application Router
   SPA routing, keyboard shortcuts, toast system, modal system
   ============================================================ */

// ── Page Config ─────────────────────────────────────────────
const pages = {
    dashboard: {
        title: 'Dashboard',
        subtitle: 'Visão geral do processamento',
        render: renderDashboard,
    },
    upload: {
        title: 'Upload de Documentos',
        subtitle: 'Enviar documentos para processamento com GLiNER 2',
        render: renderUpload,
    },
    validation: {
        title: 'Estação de Validação',
        subtitle: 'Revise e valide os dados extraídos pela IA',
        render: renderValidation,
    },
    clients: {
        title: 'Gestão de Clientes',
        subtitle: 'Cadastro e configuração das empresas atendidas',
        render: renderClients,
    },
    exports: {
        title: 'Exportações',
        subtitle: 'Histórico de exportações e integrações',
        render: renderExportsPage,
    },
    settings: {
        title: 'Configurações',
        subtitle: 'Configurações do sistema e integrações',
        render: renderSettingsPage,
    },
};

let currentPage = 'dashboard';

// ── SPA Router ──────────────────────────────────────────────
function navigateTo(page) {
    if (!pages[page]) return;

    currentPage = page;

    // Update nav
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === page);
    });

    // Update header
    document.getElementById('page-title').textContent = pages[page].title;
    document.getElementById('page-subtitle').textContent = pages[page].subtitle;

    // Render page
    const contentBody = document.getElementById('content-body');
    contentBody.scrollTop = 0;
    contentBody.innerHTML = pages[page].render();

    // Update URL hash
    window.location.hash = page;
}

// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Nav click handlers
    document.querySelectorAll('.nav-item[data-page]').forEach(item => {
        item.addEventListener('click', () => navigateTo(item.dataset.page));
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                navigateTo(item.dataset.page);
            }
        });
    });

    // Sidebar toggle
    document.getElementById('btn-toggle-sidebar').addEventListener('click', toggleSidebar);

    // Handle hash routing
    const hash = window.location.hash.substring(1);
    if (hash && pages[hash]) {
        navigateTo(hash);
    } else {
        navigateTo('dashboard');
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);

    // Global search focus
    document.getElementById('global-search').addEventListener('focus', function () {
        this.select();
    });
});

// ── Sidebar Toggle ──────────────────────────────────────────
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('collapsed');
}

// ── Keyboard Shortcuts ──────────────────────────────────────
function handleKeyboard(e) {
    // Ctrl+B = toggle sidebar
    if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
        return;
    }

    // Ctrl+K = focus search
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        document.getElementById('global-search').focus();
        return;
    }

    // Ctrl+Enter = validate (in validation page)
    if (e.ctrlKey && e.key === 'Enter' && currentPage === 'validation') {
        e.preventDefault();
        if (typeof validateDocument === 'function') validateDocument();
        return;
    }

    // Ctrl+ArrowRight = next document
    if (e.ctrlKey && e.key === 'ArrowRight' && currentPage === 'validation') {
        e.preventDefault();
        if (typeof nextDocument === 'function') nextDocument();
        return;
    }

    // Ctrl+ArrowLeft = prev document
    if (e.ctrlKey && e.key === 'ArrowLeft' && currentPage === 'validation') {
        e.preventDefault();
        if (typeof prevDocument === 'function') prevDocument();
        return;
    }

    // Escape = close modal
    if (e.key === 'Escape') {
        closeModal();
        return;
    }

    // Number shortcuts for navigation (Alt+1, Alt+2, etc.)
    if (e.altKey && e.key >= '1' && e.key <= '6') {
        e.preventDefault();
        const pageKeys = Object.keys(pages);
        const idx = parseInt(e.key) - 1;
        if (idx < pageKeys.length) {
            navigateTo(pageKeys[idx]);
        }
    }
}

// ── Toast System ────────────────────────────────────────────
function showToast(message, type = 'success', duration = 4000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        success: '✅',
        warning: '⚠️',
        error: '❌',
        info: 'ℹ️',
    };

    toast.innerHTML = `
    <span>${icons[type] || '📌'}</span>
    <span style="flex:1">${message}</span>
    <button class="btn-ghost" onclick="this.parentElement.remove()" style="padding:0;border:none;background:none;cursor:pointer;color:var(--text-tertiary);font-size:14px">✕</button>
  `;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(60px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ── Modal System ────────────────────────────────────────────
function showModal(title, bodyHtml, footerHtml = '') {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = bodyHtml;
    document.getElementById('modal-footer').innerHTML = footerHtml;
    document.getElementById('modal-overlay').classList.add('active');
}

function closeModal() {
    document.getElementById('modal-overlay').classList.remove('active');
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.id === 'modal-overlay') {
        closeModal();
    }
});

// ── Placeholder Pages ───────────────────────────────────────
function renderExportsPage() {
    return `
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">Histórico de Exportações</div>
          <div class="card-subtitle">Últimas exportações e integrações com ERPs</div>
        </div>
        <button class="btn btn-primary" onclick="showToast('Nova exportação iniciada', 'info')">📥 Nova Exportação</button>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Data</th>
              <th>Cliente</th>
              <th>Tipo</th>
              <th>Documentos</th>
              <th>Formato</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="font-variant-numeric:tabular-nums">08/03/2026 18:00</td>
              <td>Construtora Horizonte</td>
              <td>Notas Fiscais</td>
              <td>45</td>
              <td><span class="badge badge-info">JSON</span></td>
              <td><span class="badge badge-success">Concluído</span></td>
              <td><button class="btn btn-sm btn-ghost">⬇️</button></td>
            </tr>
            <tr>
              <td style="font-variant-numeric:tabular-nums">08/03/2026 15:30</td>
              <td>Farmácia Popular</td>
              <td>Guias DARF</td>
              <td>12</td>
              <td><span class="badge badge-neutral">CSV</span></td>
              <td><span class="badge badge-success">Concluído</span></td>
              <td><button class="btn btn-sm btn-ghost">⬇️</button></td>
            </tr>
            <tr>
              <td style="font-variant-numeric:tabular-nums">07/03/2026 20:00</td>
              <td>Supermercado Bom Preço</td>
              <td>Lote Completo</td>
              <td>128</td>
              <td><span class="badge badge-info">API (Domínio)</span></td>
              <td><span class="badge badge-success">Concluído</span></td>
              <td><button class="btn btn-sm btn-ghost">🔗</button></td>
            </tr>
            <tr>
              <td style="font-variant-numeric:tabular-nums">07/03/2026 14:00</td>
              <td>Tech Solutions</td>
              <td>Comp. PIX + Boletos</td>
              <td>34</td>
              <td><span class="badge badge-neutral">XLSX</span></td>
              <td><span class="badge badge-success">Concluído</span></td>
              <td><button class="btn btn-sm btn-ghost">⬇️</button></td>
            </tr>
            <tr>
              <td style="font-variant-numeric:tabular-nums">06/03/2026 19:00</td>
              <td>Restaurante Sabor & Arte</td>
              <td>Notas Fiscais</td>
              <td>67</td>
              <td><span class="badge badge-info">API (Alterdata)</span></td>
              <td><span class="badge badge-warning">Parcial</span></td>
              <td><button class="btn btn-sm btn-ghost">🔄</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function renderSettingsPage() {
    return `
    <div class="grid-2">
      <!-- GLiNER 2 Config -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🧠 Motor de IA — GLiNER 2</div>
            <div class="card-subtitle">Configurações do modelo de extração</div>
          </div>
          <span class="badge badge-success">Online</span>
        </div>
        <div class="form-group">
          <label class="form-label">Modelo</label>
          <select class="form-input">
            <option selected>fastino/gliner2-base-v1 (205M params)</option>
            <option>fastino/gliner2-large-v1 (340M params)</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Threshold de Confiança Global</label>
          <input type="range" min="50" max="99" value="80" class="form-input" style="padding:0;height:auto"
                 oninput="this.nextElementSibling.textContent=this.value+'%'" />
          <span style="font-size:var(--text-sm);font-weight:600">80%</span>
        </div>
        <div class="form-group">
          <label class="form-label">Batch Size</label>
          <input type="number" class="form-input" value="8" min="1" max="32" />
          <div class="form-hint">Documentos processados por lote (recomendado: 8 para CPU)</div>
        </div>
        <div class="form-group">
          <label class="form-label">Dispositivo</label>
          <select class="form-input">
            <option selected>CPU (Recomendado)</option>
            <option>CUDA (GPU NVIDIA)</option>
          </select>
        </div>
        <button class="btn btn-sm btn-secondary" onclick="showToast('Configurações do modelo salvas!','success')">💾 Salvar</button>
      </div>

      <!-- OCR Config -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">📸 Motor de OCR</div>
            <div class="card-subtitle">Configurações de leitura óptica</div>
          </div>
          <span class="badge badge-success">Ativo</span>
        </div>
        <div class="form-group">
          <label class="form-label">Engine de OCR</label>
          <select class="form-input">
            <option selected>PaddleOCR (Recomendado)</option>
            <option>Tesseract OCR</option>
            <option>EasyOCR</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">Idiomas</label>
          <input type="text" class="form-input" value="pt, en" />
          <div class="form-hint">Idiomas para reconhecimento de texto</div>
        </div>
        <div class="form-group">
          <label class="form-label">Pré-processamento de Imagem</label>
          <div style="display:flex;flex-direction:column;gap:var(--space-2)">
            <label style="display:flex;align-items:center;gap:var(--space-2);font-size:var(--text-sm);cursor:pointer">
              <input type="checkbox" checked /> Correção de rotação automática
            </label>
            <label style="display:flex;align-items:center;gap:var(--space-2);font-size:var(--text-sm);cursor:pointer">
              <input type="checkbox" checked /> Binarização adaptativa
            </label>
            <label style="display:flex;align-items:center;gap:var(--space-2);font-size:var(--text-sm);cursor:pointer">
              <input type="checkbox" checked /> Remoção de ruído
            </label>
            <label style="display:flex;align-items:center;gap:var(--space-2);font-size:var(--text-sm);cursor:pointer">
              <input type="checkbox" /> Resolução super-resolução (lento)
            </label>
          </div>
        </div>
        <button class="btn btn-sm btn-secondary" onclick="showToast('Configurações OCR salvas!','success')">💾 Salvar</button>
      </div>

      <!-- Security -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔒 Privacidade & LGPD</div>
            <div class="card-subtitle">Segurança e conformidade de dados</div>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:var(--space-4)">
          <div style="display:flex;align-items:center;gap:var(--space-3);padding:var(--space-3);background:var(--color-success-bg);border:1px solid var(--color-success-border);border-radius:var(--border-radius-sm)">
            <span>✅</span>
            <div>
              <div style="font-size:var(--text-sm);font-weight:600;color:var(--color-success)">Processamento 100% Local</div>
              <div style="font-size:var(--text-xs);color:var(--text-tertiary)">Nenhum dado é enviado para APIs externas</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:var(--space-3);padding:var(--space-3);background:var(--color-success-bg);border:1px solid var(--color-success-border);border-radius:var(--border-radius-sm)">
            <span>✅</span>
            <div>
              <div style="font-size:var(--text-sm);font-weight:600;color:var(--color-success)">Modelo GLiNER 2 Offline</div>
              <div style="font-size:var(--text-xs);color:var(--text-tertiary)">IA roda sem conexão com internet</div>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:var(--space-3);padding:var(--space-3);background:var(--color-success-bg);border:1px solid var(--color-success-border);border-radius:var(--border-radius-sm)">
            <span>✅</span>
            <div>
              <div style="font-size:var(--text-sm);font-weight:600;color:var(--color-success)">Dados Criptografados em Repouso</div>
              <div style="font-size:var(--text-xs);color:var(--text-tertiary)">AES-256 para documentos armazenados</div>
            </div>
          </div>
        </div>
        <div class="form-group" style="margin-top:var(--space-5)">
          <label class="form-label">Retenção de Documentos</label>
          <select class="form-input">
            <option>30 dias</option>
            <option selected>90 dias</option>
            <option>180 dias</option>
            <option>365 dias</option>
            <option>Sem limite</option>
          </select>
        </div>
      </div>

      <!-- Notifications -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">🔔 Notificações</div>
            <div class="card-subtitle">Alertas e comunicações</div>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:var(--space-3)">
          <label style="display:flex;align-items:center;justify-content:space-between;font-size:var(--text-sm);cursor:pointer;padding:var(--space-2) 0">
            <span>Email ao completar lote</span>
            <input type="checkbox" checked />
          </label>
          <label style="display:flex;align-items:center;justify-content:space-between;font-size:var(--text-sm);cursor:pointer;padding:var(--space-2) 0">
            <span>Alerta de erros de OCR</span>
            <input type="checkbox" checked />
          </label>
          <label style="display:flex;align-items:center;justify-content:space-between;font-size:var(--text-sm);cursor:pointer;padding:var(--space-2) 0">
            <span>Resumo diário por email</span>
            <input type="checkbox" />
          </label>
          <label style="display:flex;align-items:center;justify-content:space-between;font-size:var(--text-sm);cursor:pointer;padding:var(--space-2) 0">
            <span>Webhook para integrações</span>
            <input type="checkbox" checked />
          </label>
        </div>
        <div class="form-group" style="margin-top:var(--space-4)">
          <label class="form-label">Email de Notificações</label>
          <input type="email" class="form-input" value="ricardo@contabilidade.com.br" />
        </div>
        <button class="btn btn-sm btn-secondary" onclick="showToast('Notificações atualizadas!','success')">💾 Salvar</button>
      </div>
    </div>
  `;
}
