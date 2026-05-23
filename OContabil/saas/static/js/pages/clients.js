/* ============================================================
   ContaDoc AI — Clients Management Page
   ============================================================ */

function renderClients() {
    const clients = MockData.clients;

    return `
    <!-- Header Actions -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:var(--space-6)">
      <div style="display:flex;gap:var(--space-3)">
        <div class="header-search" style="position:relative">
          <span class="search-icon" style="position:absolute;left:10px;top:50%;transform:translateY(-50%);font-size:14px;pointer-events:none">🔍</span>
          <input type="text" 
                 class="form-input" 
                 placeholder="Buscar cliente..." 
                 style="padding-left:36px;width:280px;height:38px"
                 oninput="filterClients(this.value)" />
        </div>
        <select class="form-input" style="width:180px;height:38px">
          <option>Todos os status</option>
          <option>Com pendências</option>
          <option>Em dia</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="showAddClientModal()">
        ➕ Novo Cliente
      </button>
    </div>

    <!-- Summary Stats -->
    <div class="stats-grid" style="margin-bottom:var(--space-6)">
      <div class="stat-card" style="--stat-accent: var(--color-accent)">
        <div class="stat-icon" style="background:var(--color-accent-glow);color:var(--color-accent-light)">🏢</div>
        <div class="stat-content">
          <div class="stat-value">${clients.length}</div>
          <div class="stat-label">Clientes ativos</div>
        </div>
      </div>
      <div class="stat-card" style="--stat-accent: var(--color-success)">
        <div class="stat-icon" style="background:var(--color-success-bg);color:var(--color-success)">📄</div>
        <div class="stat-content">
          <div class="stat-value">${clients.reduce((s, c) => s + c.docsMonth, 0).toLocaleString('pt-BR')}</div>
          <div class="stat-label">Documentos no mês</div>
        </div>
      </div>
      <div class="stat-card" style="--stat-accent: var(--color-warning)">
        <div class="stat-icon" style="background:var(--color-warning-bg);color:var(--color-warning)">⏳</div>
        <div class="stat-content">
          <div class="stat-value">${clients.reduce((s, c) => s + c.pendingDocs, 0)}</div>
          <div class="stat-label">Documentos pendentes</div>
        </div>
      </div>
    </div>

    <!-- Clients Grid -->
    <div class="clients-grid" id="clients-grid">
      ${clients.map(c => renderClientCard(c)).join('')}
    </div>
  `;
}

function renderClientCard(client) {
    const initials = client.name.split(' ').slice(0, 2).map(w => w[0]).join('');
    const validatedPercent = client.docsMonth > 0
        ? Math.round((client.docsValidated / client.docsMonth) * 100)
        : 100;

    return `
    <div class="client-card" onclick="showClientDetail(${client.id})" data-client-id="${client.id}">
      <div class="client-card-header">
        <div class="client-avatar" style="background:${client.color}">${initials}</div>
        <div style="flex:1;min-width:0">
          <div class="client-name">${client.name}</div>
          <div class="client-cnpj">${client.cnpj}</div>
        </div>
        ${client.pendingDocs > 0
            ? `<span class="badge badge-warning">${client.pendingDocs} pendentes</span>`
            : '<span class="badge badge-success">Em dia</span>'
        }
      </div>
      
      <!-- Progress Bar -->
      <div style="margin-bottom:var(--space-4)">
        <div style="display:flex;justify-content:space-between;margin-bottom:var(--space-1)">
          <span style="font-size:var(--text-xs);color:var(--text-tertiary)">Progresso</span>
          <span style="font-size:var(--text-xs);font-weight:600;color:var(--text-primary)">${validatedPercent}%</span>
        </div>
        <div style="height:4px;background:var(--surface-elevated);border-radius:2px;overflow:hidden">
          <div style="height:100%;width:${validatedPercent}%;background:${validatedPercent === 100 ? 'var(--color-success)' : 'var(--color-accent)'};border-radius:2px;transition:width 0.5s"></div>
        </div>
      </div>

      <div class="client-stats">
        <div class="client-stat">
          <div class="client-stat-value">${client.docsMonth.toLocaleString('pt-BR')}</div>
          <div class="client-stat-label">Docs/Mês</div>
        </div>
        <div class="client-stat">
          <div class="client-stat-value">${client.docsValidated.toLocaleString('pt-BR')}</div>
          <div class="client-stat-label">Validados</div>
        </div>
        <div class="client-stat">
          <div class="client-stat-value" style="color:${client.pendingDocs > 0 ? 'var(--color-warning)' : 'var(--color-success)'}">${client.pendingDocs}</div>
          <div class="client-stat-label">Pendentes</div>
        </div>
      </div>
    </div>
  `;
}

function filterClients(query) {
    const cards = document.querySelectorAll('.client-card');
    const q = query.toLowerCase();

    cards.forEach(card => {
        const name = card.querySelector('.client-name').textContent.toLowerCase();
        const cnpj = card.querySelector('.client-cnpj').textContent;

        if (name.includes(q) || cnpj.includes(q)) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    });
}

function showClientDetail(clientId) {
    const client = MockData.clients.find(c => c.id === clientId);
    if (!client) return;

    const initials = client.name.split(' ').slice(0, 2).map(w => w[0]).join('');

    showModal(client.name, `
    <div style="display:flex;align-items:center;gap:var(--space-4);margin-bottom:var(--space-6)">
      <div class="client-avatar" style="background:${client.color};width:56px;height:56px;font-size:var(--text-xl)">${initials}</div>
      <div>
        <div style="font-size:var(--text-lg);font-weight:700">${client.name}</div>
        <div style="font-family:var(--font-mono);font-size:var(--text-sm);color:var(--text-tertiary)">${client.cnpj}</div>
      </div>
    </div>

    <div class="grid-3" style="margin-bottom:var(--space-6)">
      <div style="text-align:center;padding:var(--space-4);background:var(--surface-elevated);border-radius:var(--border-radius-sm)">
        <div style="font-size:var(--text-2xl);font-weight:800;color:var(--text-primary)">${client.docsMonth.toLocaleString('pt-BR')}</div>
        <div style="font-size:var(--text-xs);color:var(--text-tertiary)">Documentos/Mês</div>
      </div>
      <div style="text-align:center;padding:var(--space-4);background:var(--surface-elevated);border-radius:var(--border-radius-sm)">
        <div style="font-size:var(--text-2xl);font-weight:800;color:var(--color-success)">${client.docsValidated.toLocaleString('pt-BR')}</div>
        <div style="font-size:var(--text-xs);color:var(--text-tertiary)">Validados</div>
      </div>
      <div style="text-align:center;padding:var(--space-4);background:var(--surface-elevated);border-radius:var(--border-radius-sm)">
        <div style="font-size:var(--text-2xl);font-weight:800;color:${client.pendingDocs > 0 ? 'var(--color-warning)' : 'var(--color-success)'}">${client.pendingDocs}</div>
        <div style="font-size:var(--text-xs);color:var(--text-tertiary)">Pendentes</div>
      </div>
    </div>

    <div class="divider"></div>

    <div class="form-group">
      <label class="form-label">Regime Tributário</label>
      <select class="form-input">
        <option>Simples Nacional</option>
        <option>Lucro Presumido</option>
        <option selected>Lucro Real</option>
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Schema de Extração Padrão</label>
      <select class="form-input">
        <option value="auto" selected>Detecção Automática (GLiNER 2)</option>
        ${Object.entries(MockData.extractionSchemas).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('')}
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Pasta de Destino</label>
      <input type="text" class="form-input" value="/documentos/${client.cnpj.replace(/[./-]/g, '')}/" />
    </div>
    <div class="form-group">
      <label class="form-label">Integração ERP</label>
      <div style="display:flex;gap:var(--space-3)">
        <select class="form-input" style="flex:1">
          <option>Sem integração</option>
          <option selected>Domínio Sistemas</option>
          <option>Alterdata</option>
          <option>Fortes</option>
          <option>Questor</option>
        </select>
        <button class="btn btn-sm btn-secondary">🔗 Testar</button>
      </div>
    </div>
  `, `
    <button class="btn btn-secondary" onclick="closeModal()">Fechar</button>
    <button class="btn btn-primary" onclick="closeModal(); showToast('Configurações salvas!', 'success')">💾 Salvar</button>
  `);
}

function showAddClientModal() {
    showModal('Novo Cliente', `
    <div class="form-group">
      <label class="form-label">CNPJ</label>
      <input type="text" class="form-input" placeholder="00.000.000/0001-00" id="new-client-cnpj" />
      <div class="form-hint">A razão social será buscada automaticamente via CNPJ</div>
    </div>
    <div class="form-group">
      <label class="form-label">Razão Social</label>
      <input type="text" class="form-input" placeholder="Nome da empresa" id="new-client-name" />
    </div>
    <div class="form-group">
      <label class="form-label">Regime Tributário</label>
      <select class="form-input">
        <option>Simples Nacional</option>
        <option>Lucro Presumido</option>
        <option>Lucro Real</option>
        <option>MEI</option>
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Contato (Email)</label>
      <input type="email" class="form-input" placeholder="financeiro@empresa.com.br" />
    </div>
  `, `
    <button class="btn btn-secondary" onclick="closeModal()">Cancelar</button>
    <button class="btn btn-primary" onclick="closeModal(); showToast('Cliente adicionado com sucesso!', 'success')">➕ Adicionar</button>
  `);
}
