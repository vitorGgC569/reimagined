/* ============================================================
   ContaDoc AI — Dashboard Page
   ============================================================ */

function renderDashboard() {
    const stats = MockData.stats;
    const chart = MockData.volumeChart;
    const activity = MockData.recentActivity;
    const maxCount = Math.max(...chart.map(d => d.count));

    return `
    <!-- Stat Cards -->
    <div class="stats-grid">
      <div class="stat-card" style="--stat-accent: var(--color-accent)">
        <div class="stat-icon" style="background: var(--color-accent-glow); color: var(--color-accent-light)">📄</div>
        <div class="stat-content">
          <div class="stat-value">${stats.processedToday.toLocaleString('pt-BR')}</div>
          <div class="stat-label">Processados hoje</div>
          <span class="stat-trend up">↑ ${stats.processedTodayTrend}</span>
        </div>
      </div>

      <div class="stat-card" style="--stat-accent: var(--color-warning)">
        <div class="stat-icon" style="background: var(--color-warning-bg); color: var(--color-warning)">⏳</div>
        <div class="stat-content">
          <div class="stat-value">${stats.awaitingValidation}</div>
          <div class="stat-label">Aguardando validação</div>
          <span class="stat-trend down">↓ ${stats.awaitingValidationTrend}</span>
        </div>
      </div>

      <div class="stat-card" style="--stat-accent: var(--color-error)">
        <div class="stat-icon" style="background: var(--color-error-bg); color: var(--color-error)">⚠️</div>
        <div class="stat-content">
          <div class="stat-value">${stats.readingErrors}</div>
          <div class="stat-label">Erros de leitura</div>
        </div>
      </div>

      <div class="stat-card" style="--stat-accent: var(--color-success)">
        <div class="stat-icon" style="background: var(--color-success-bg); color: var(--color-success)">✓</div>
        <div class="stat-content">
          <div class="stat-value">${stats.accuracyRate}%</div>
          <div class="stat-label">Acurácia GLiNER 2</div>
          <span class="stat-trend up">↑ +0.8%</span>
        </div>
      </div>
    </div>

    <!-- Charts & Activity Row -->
    <div class="grid-2">
      <!-- Volume Chart -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Volume de Processamento</div>
            <div class="card-subtitle">Últimos 14 dias · ${stats.totalDocumentsMonth.toLocaleString('pt-BR')} docs/mês</div>
          </div>
          <div style="display:flex;gap:var(--space-2)">
            <span class="badge badge-info">Tempo médio: ${stats.avgProcessingTime}</span>
          </div>
        </div>
        <div class="chart-container">
          ${chart.map(d => `
            <div class="chart-bar" style="height: ${(d.count / maxCount * 100)}%">
              <div class="chart-tooltip">${d.day}: ${d.count} docs</div>
            </div>
          `).join('')}
        </div>
        <div class="chart-labels">
          ${chart.map(d => `<div class="chart-label">${d.day}</div>`).join('')}
        </div>
      </div>

      <!-- Recent Activity -->
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Atividade Recente</div>
            <div class="card-subtitle">Último processamento: hoje às 18:30</div>
          </div>
          <button class="btn btn-sm btn-ghost">Ver tudo →</button>
        </div>
        <div style="display:flex;flex-direction:column;gap:2px">
          ${activity.map(a => `
            <div style="display:flex;align-items:center;gap:var(--space-3);padding:var(--space-2) var(--space-3);border-radius:var(--border-radius-sm);transition:background 0.15s" onmouseover="this.style.background='var(--surface-card-hover)'" onmouseout="this.style.background='transparent'">
              <span style="font-size:var(--text-xs);color:var(--text-tertiary);font-variant-numeric:tabular-nums;min-width:40px">${a.time}</span>
              <span class="badge badge-${a.status}" style="min-width:90px">${a.action}</span>
              <span style="font-size:var(--text-sm);color:var(--text-secondary);flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${a.client}</span>
              <span style="font-size:var(--text-xs);color:var(--text-tertiary);font-family:var(--font-mono);white-space:nowrap">${a.doc}</span>
            </div>
          `).join('')}
        </div>
      </div>
    </div>

    <!-- Top Clients -->
    <div class="card" style="margin-top:var(--space-5)">
      <div class="card-header">
        <div>
          <div class="card-title">Clientes por Volume</div>
          <div class="card-subtitle">${stats.totalClients} clientes ativos</div>
        </div>
        <button class="btn btn-sm btn-secondary" onclick="navigateTo('clients')">Ver todos →</button>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Cliente</th>
              <th>CNPJ</th>
              <th>Docs/Mês</th>
              <th>Validados</th>
              <th>Pendentes</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${MockData.clients.slice(0, 5).map(c => `
              <tr>
                <td>
                  <div style="display:flex;align-items:center;gap:var(--space-3)">
                    <div class="client-avatar" style="width:32px;height:32px;font-size:var(--text-xs);background:${c.color}">${c.name.split(' ').slice(0, 2).map(w => w[0]).join('')}</div>
                    <span style="font-weight:600">${c.name}</span>
                  </div>
                </td>
                <td style="font-family:var(--font-mono);font-size:var(--text-xs)">${c.cnpj}</td>
                <td style="font-weight:600;font-variant-numeric:tabular-nums">${c.docsMonth.toLocaleString('pt-BR')}</td>
                <td style="font-variant-numeric:tabular-nums">${c.docsValidated.toLocaleString('pt-BR')}</td>
                <td style="font-variant-numeric:tabular-nums">${c.pendingDocs}</td>
                <td>
                  ${c.pendingDocs === 0
            ? '<span class="badge badge-success">Em dia</span>'
            : c.pendingDocs > 15
                ? '<span class="badge badge-warning">Pendente</span>'
                : '<span class="badge badge-info">Processando</span>'
        }
                </td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    </div>
  `;
}
