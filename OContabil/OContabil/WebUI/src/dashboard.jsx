/* ============================================================
   OContabil — Painel (Dashboard)
   ============================================================ */

function KpiCard({ label, value, unit, sub, icon, trend, accent }) {
  return (
    <Panel style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 12, position: 'relative', overflow: 'hidden' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12.5, color: 'var(--text-2)', fontWeight: 500 }}>{label}</span>
        <span style={{ color: accent ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex', padding: 6, borderRadius: 6, background: accent ? 'var(--accent-weak)' : 'var(--subtle)' }}>
          <Icon name={icon} size={16} />
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span className="mono" style={{ fontSize: 30, fontWeight: 600, letterSpacing: '-0.02em', lineHeight: 1, color: 'var(--text)' }}>{value}</span>
        {unit && <span className="mono" style={{ fontSize: 15, color: 'var(--text-2)', fontWeight: 500 }}>{unit}</span>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
        {trend && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, fontSize: 11.5, fontWeight: 500, color: trend.dir === 'up' ? 'var(--st-approved)' : 'var(--st-rejected)' }} className="mono">
            {trend.dir === 'up' ? '▲' : '▼'} {trend.value}
          </span>
        )}
        <span style={{ fontSize: 11.5, color: 'var(--text-3)' }}>{sub}</span>
      </div>
    </Panel>
  );
}

function StatusDonut({ dist }) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0);
  const segs = [
    { key: 'aprovado', label: 'Processado / aprovado', val: dist.aprovado, color: 'var(--st-approved)' },
    { key: 'revisar', label: 'Em revisão', val: dist.revisar, color: 'var(--st-review)' },
    { key: 'rejeitado', label: 'Rejeitado', val: dist.rejeitado, color: 'var(--st-rejected)' },
    { key: 'pendente', label: 'Pendente / processando', val: dist.pendente, color: 'var(--st-pending)' },
  ];
  const R = 54, C = 2 * Math.PI * R;
  let offset = 0;
  const [hover, setHover] = useState(null);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ position: 'relative', width: 140, height: 140, flexShrink: 0 }}>
        <svg viewBox="0 0 140 140" style={{ transform: 'rotate(-90deg)' }}>
          <circle cx="70" cy="70" r={R} fill="none" stroke="var(--subtle)" strokeWidth="16" />
          {segs.map((s) => {
            const frac = s.val / total;
            const dash = frac * C;
            const el = (
              <circle key={s.key} cx="70" cy="70" r={R} fill="none" stroke={s.color}
                strokeWidth={hover === s.key ? 19 : 16} strokeDasharray={`${dash} ${C - dash}`} strokeDashoffset={-offset}
                style={{ transition: 'stroke-width .15s', cursor: 'pointer' }}
                onMouseEnter={() => setHover(s.key)} onMouseLeave={() => setHover(null)} />
            );
            offset += dash;
            return el;
          })}
        </svg>
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <span className="mono" style={{ fontSize: 26, fontWeight: 600, lineHeight: 1, color: 'var(--text)' }}>{hover ? dist[hover] : total}</span>
          <span style={{ fontSize: 10.5, color: 'var(--text-2)', marginTop: 3 }}>{hover ? segs.find(s => s.key === hover).label.split(' ')[0] : 'documentos'}</span>
        </div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 11, flex: 1, minWidth: 180 }}>
        {segs.map(s => (
          <div key={s.key} onMouseEnter={() => setHover(s.key)} onMouseLeave={() => setHover(null)}
            style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', opacity: hover && hover !== s.key ? 0.5 : 1, transition: 'opacity .15s' }}>
            <span style={{ width: 9, height: 9, borderRadius: 3, background: s.color, flexShrink: 0 }} />
            <span style={{ fontSize: 12.5, color: 'var(--text-2)', flex: 1 }}>{s.label}</span>
            <span className="mono" style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>{s.val}</span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)', width: 38, textAlign: 'right' }}>{Math.round(s.val / total * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Dashboard({ nav, openDoc }) {
  const DB = window.DB;
  const k = DB.kpis;
  const parados = DB.documents.filter(d => d.diasParado != null).sort((a, b) => b.diasParado - a.diasParado);
  const [bannerOpen, setBannerOpen] = useState(true);
  const maxVol = Math.max(...DB.topClientes.map(c => c.volume));

  return (
    <div style={{ animation: 'om-fade-in .25s ease' }}>
      <SectionHeader title="Painel" subtitle="Terça-feira, 1 de junho de 2026 · Visão geral do processamento"
        action={
          <div style={{ display: 'flex', gap: 10 }}>
            <Button variant="default" icon="calendario" size="md">Últimos 14 dias</Button>
            <Button variant="primary" icon="upload" size="md" onClick={() => nav('documentos')}>Importar documentos</Button>
          </div>
        } />

      {/* Banner de alerta âmbar */}
      {bannerOpen && parados.length > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14, padding: '13px 16px', marginBottom: 18,
          background: 'var(--st-review-bg)', border: '1px solid ' + 'var(--st-review)' + '40', borderRadius: 'var(--r-md)',
          animation: 'om-pop-in .25s ease',
        }}>
          <span style={{ color: 'var(--st-review)', display: 'inline-flex' }}><Icon name="alerta" size={20} /></span>
          <div style={{ flex: 1 }}>
            <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--st-review)' }}>{parados.length} documentos parados há mais de 3 dias</span>
            <span style={{ fontSize: 13, color: 'var(--text-2)', marginLeft: 8 }}>
              aguardando revisão ou pendentes — limite configurado em 3 dias.
            </span>
          </div>
          <Button variant="default" size="sm" onClick={() => nav('documentos', { filtro: 'parados' })} iconRight="setaDir">Revisar agora</Button>
          <button onClick={() => setBannerOpen(false)} aria-label="Dispensar" style={{ background: 'transparent', border: 'none', color: 'var(--st-review)', padding: 5, display: 'inline-flex', borderRadius: 5 }}><Icon name="x" size={16} /></button>
        </div>
      )}

      {/* KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 14, marginBottom: 16 }}>
        <KpiCard label="Processados hoje" value={k.processadosHoje} icon="check" accent sub="vs. ontem" trend={{ dir: 'up', value: '+12%' }} />
        <KpiCard label="Na fila" value={k.naFila} icon="relogio" sub="em processamento" />
        <KpiCard label="Aguardando revisão" value={k.aguardandoRevisao} icon="olho" sub={parados.length + ' parados >3 dias'} />
        <KpiCard label="Confiança média" value={k.confiancaMedia.toFixed(1)} unit="%" icon="escudo" accent sub="últimos 14 dias" trend={{ dir: 'up', value: '+1,4 pts' }} />
      </div>

      {/* Linha de gráficos */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 1fr', gap: 14, marginBottom: 16 }}>
        <Panel style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 18 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Confiança da extração</h3>
              <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-2)' }}>Média diária · últimos 14 dias</p>
            </div>
            <span className="mono" style={{ fontSize: 13, color: 'var(--st-approved)', fontWeight: 500, background: 'var(--st-approved-bg)', padding: '4px 9px', borderRadius: 5 }}>91,4% atual</span>
          </div>
          <AreaChart data={DB.confSeries} valueKey="conf" min={84} max={96} height={150} format={v => v.toFixed(1) + '%'} />
        </Panel>

        <Panel style={{ padding: 20 }}>
          <div style={{ marginBottom: 18 }}>
            <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Status dos documentos</h3>
            <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-2)' }}>Distribuição atual da base</p>
          </div>
          <StatusDonut dist={DB.statusDist} />
        </Panel>
      </div>

      {/* Linha inferior: top clientes + parados */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.25fr', gap: 14 }}>
        <Panel style={{ padding: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
            <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Top clientes por volume</h3>
            <button onClick={() => nav('clientes')} style={{ background: 'none', border: 'none', color: 'var(--accent)', fontSize: 12.5, fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 3 }}>Ver todos <Icon name="chevRight" size={13} /></button>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {DB.topClientes.map((c, i) => (
              <div key={c.id} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)', width: 14 }}>{i + 1}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span style={{ fontSize: 12.5, color: 'var(--text)', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 180 }}>{c.fantasia}</span>
                    <span className="mono" style={{ fontSize: 12.5, color: 'var(--text-2)' }}>{c.volume}</span>
                  </div>
                  <div style={{ height: 6, background: 'var(--subtle)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: (c.volume / maxVol * 100) + '%', background: 'var(--accent)', borderRadius: 999, opacity: 1 - i * 0.1 }} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid var(--hairline)' }}>
            <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Documentos parados</h3>
            <span style={{ fontSize: 12, color: 'var(--st-review)', fontWeight: 500 }}>{parados.length} itens</span>
          </div>
          <div>
            {parados.slice(0, 5).map((d, i) => (
              <button key={d.id} onClick={() => openDoc(d)} style={{
                display: 'grid', gridTemplateColumns: '28px 1fr auto auto', alignItems: 'center', gap: 12, width: '100%',
                padding: '11px 20px', background: 'transparent', border: 'none', borderTop: i ? '1px solid var(--hairline-2)' : 'none', textAlign: 'left',
              }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--zebra)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <span style={{ color: 'var(--text-3)' }}><Icon name={window.DOC_ICON[d.tipo] || 'arquivo'} size={17} /></span>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, color: 'var(--text)', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{d.tipo} {d.numero} · {d.cliente}</div>
                  <div className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>R$ {DB.brl(d.valor)}</div>
                </div>
                <span className="mono" style={{ fontSize: 11.5, color: 'var(--st-review)', fontWeight: 500 }}>{d.diasParado}d parado</span>
                <StatusBadge status={d.status} size="sm" />
              </button>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

window.Dashboard = Dashboard;
