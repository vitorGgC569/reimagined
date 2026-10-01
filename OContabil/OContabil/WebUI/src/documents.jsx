/* ============================================================
   OContabil — Documentos (lista densa + filtros)
   ============================================================ */

function FilterChip({ label, value, options, onChange, icon }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', h);
    return () => document.removeEventListener('mousedown', h);
  }, []);
  const active = value && value !== 'todos';
  const cur = options.find(o => o.value === value);
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        display: 'inline-flex', alignItems: 'center', gap: 7, height: 34, padding: '0 11px',
        fontSize: 13, fontWeight: 500, borderRadius: 'var(--r-md)',
        background: active ? 'var(--accent-weak)' : 'var(--surface)',
        border: '1px solid ' + (active ? 'var(--accent-weak-2)' : 'var(--hairline)'),
        color: active ? 'var(--accent-ink)' : 'var(--text-2)',
      }}>
        {icon && <Icon name={icon} size={14} />}
        <span>{label}{active && cur ? ': ' : ''}</span>
        {active && cur && <span style={{ color: 'var(--text)', fontWeight: 600 }}>{cur.label}</span>}
        <Icon name="chevDown" size={13} />
      </button>
      {open && (
        <div style={{ position: 'absolute', top: 'calc(100% + 6px)', left: 0, zIndex: 50, minWidth: 190, background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', boxShadow: 'var(--shadow-2)', padding: 5, animation: 'om-pop-in .14s ease' }}>
          {options.map(o => (
            <button key={o.value} onClick={() => { onChange(o.value); setOpen(false); }} style={{
              display: 'flex', alignItems: 'center', gap: 9, width: '100%', padding: '8px 10px', fontSize: 13,
              background: o.value === value ? 'var(--subtle)' : 'transparent', border: 'none', borderRadius: 5,
              color: 'var(--text)', textAlign: 'left',
            }}
              onMouseEnter={e => { if (o.value !== value) e.currentTarget.style.background = 'var(--zebra)'; }}
              onMouseLeave={e => { if (o.value !== value) e.currentTarget.style.background = 'transparent'; }}>
              {o.dot && <span style={{ width: 8, height: 8, borderRadius: 999, background: o.dot }} />}
              <span style={{ flex: 1 }}>{o.label}</span>
              {o.value === value && <span style={{ color: 'var(--accent)' }}><Icon name="check" size={14} /></span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function DocumentsScreen({ openDoc, density, setDensity, initialFilter, toast }) {
  const DB = window.DB;
  const [search, setSearch] = useState('');
  const [fCliente, setFCliente] = useState('todos');
  const [fTipo, setFTipo] = useState('todos');
  const [fStatus, setFStatus] = useState(initialFilter === 'parados' ? 'parados' : 'todos');
  const [fConf, setFConf] = useState('todos');
  const [fValor, setFValor] = useState('todos');
  const [fPeriodo, setFPeriodo] = useState('todos');
  const [orgOpen, setOrgOpen] = useState(false);
  const [sel, setSel] = useState(new Set());
  const [sort, setSort] = useState({ key: 'data', dir: 'desc' });

  useEffect(() => { if (initialFilter === 'parados') setFStatus('parados'); }, [initialFilter]);

  const clienteOpts = [{ value: 'todos', label: 'Todos os clientes' }, ...DB.clients.map(c => ({ value: c.id, label: c.fantasia }))];
  const tipoOpts = [{ value: 'todos', label: 'Todos os tipos' }, ...DB.docTypes.map(t => ({ value: t, label: t }))];
  const statusOpts = [
    { value: 'todos', label: 'Todos os status' },
    { value: 'aprovado', label: 'Aprovado', dot: 'var(--st-approved)' },
    { value: 'revisar', label: 'Revisar', dot: 'var(--st-review)' },
    { value: 'rejeitado', label: 'Rejeitado', dot: 'var(--st-rejected)' },
    { value: 'pendente', label: 'Pendente', dot: 'var(--st-pending)' },
    { value: 'processando', label: 'Processando', dot: 'var(--st-info)' },
    { value: 'parados', label: 'Parados >3 dias', dot: 'var(--st-review)' },
  ];
  const confOpts = [
    { value: 'todos', label: 'Toda confiança' },
    { value: 'alta', label: 'Alta (≥85%)', dot: 'var(--conf-high)' },
    { value: 'media', label: 'Média (65–84%)', dot: 'var(--conf-mid)' },
    { value: 'baixa', label: 'Baixa (<65%)', dot: 'var(--conf-low)' },
  ];
  const valorOpts = [
    { value: 'todos', label: 'Qualquer valor' },
    { value: 'ate500', label: 'Até R$ 500' },
    { value: '500a5k', label: 'R$ 500 – 5.000' },
    { value: '5ka50k', label: 'R$ 5.000 – 50.000' },
    { value: 'acima50k', label: 'Acima de R$ 50.000' },
  ];
  const periodoOpts = [
    { value: 'todos', label: 'Qualquer período' },
    { value: 'hoje', label: 'Hoje' },
    { value: '7d', label: 'Últimos 7 dias' },
    { value: '30d', label: 'Últimos 30 dias' },
    { value: 'mes', label: 'Este mês' },
    { value: 'ano', label: 'Este ano' },
  ];

  const filtered = useMemo(() => {
    let rows = DB.documents.filter(d => {
      if (fCliente !== 'todos' && d.clienteId !== fCliente) return false;
      if (fTipo !== 'todos' && d.tipo !== fTipo) return false;
      if (fStatus === 'parados') { if (d.diasParado == null) return false; }
      else if (fStatus !== 'todos' && d.status !== fStatus) return false;
      if (fConf !== 'todos') {
        const c = d.confianca;
        if (c == null) return false;
        if (fConf === 'alta' && c < 85) return false;
        if (fConf === 'media' && (c < 65 || c >= 85)) return false;
        if (fConf === 'baixa' && c >= 65) return false;
      }
      if (fValor !== 'todos') {
        const v = d.valor || 0;
        if (fValor === 'ate500' && !(v <= 500)) return false;
        if (fValor === '500a5k' && !(v > 500 && v <= 5000)) return false;
        if (fValor === '5ka50k' && !(v > 5000 && v <= 50000)) return false;
        if (fValor === 'acima50k' && !(v > 50000)) return false;
      }
      if (fPeriodo !== 'todos' && d.data) {
        const now = new Date();
        const dt = d.data instanceof Date ? d.data : new Date(d.data);
        const days = (now - dt) / 86400000;
        if (fPeriodo === 'hoje' && dt.toDateString() !== now.toDateString()) return false;
        if (fPeriodo === '7d' && days > 7) return false;
        if (fPeriodo === '30d' && days > 30) return false;
        if (fPeriodo === 'mes' && (dt.getMonth() !== now.getMonth() || dt.getFullYear() !== now.getFullYear())) return false;
        if (fPeriodo === 'ano' && dt.getFullYear() !== now.getFullYear()) return false;
      }
      if (search) {
        const s = search.toLowerCase();
        if (!(d.numero.includes(s) || d.cliente.toLowerCase().includes(s) || d.emitente.toLowerCase().includes(s) || d.chave.includes(s.replace(/\s/g, '')) || d.tipo.toLowerCase().includes(s) || (/\d/.test(s) && String(Math.round(d.valor || 0)).includes(s.replace(/\D/g, ''))))) return false;
      }
      return true;
    });
    rows = rows.slice().sort((a, b) => {
      let r = 0;
      if (sort.key === 'data') r = a.data - b.data;
      else if (sort.key === 'valor') r = a.valor - b.valor;
      else if (sort.key === 'confianca') r = (a.confianca ?? -1) - (b.confianca ?? -1);
      else if (sort.key === 'cliente') r = a.cliente.localeCompare(b.cliente);
      return sort.dir === 'asc' ? r : -r;
    });
    return rows;
  }, [search, fCliente, fTipo, fStatus, fConf, fValor, fPeriodo, sort]);

  const activeFilters = [fCliente, fTipo, fStatus, fConf, fValor, fPeriodo].filter(v => v !== 'todos').length + (search ? 1 : 0);
  const clearAll = () => { setSearch(''); setFCliente('todos'); setFTipo('todos'); setFStatus('todos'); setFConf('todos'); setFValor('todos'); setFPeriodo('todos'); };

  const toggleSel = id => setSel(s => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const allSel = filtered.length > 0 && filtered.every(d => sel.has(d.id));
  const toggleAll = () => setSel(allSel ? new Set() : new Set(filtered.map(d => d.id)));

  const Th = ({ label, sortKey, align, width }) => (
    <th style={{ padding: '0 var(--cell-px)', height: 38, textAlign: align || 'left', width, fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.04em', whiteSpace: 'nowrap', position: 'sticky', top: 0, background: 'var(--surface-2)', borderBottom: '1px solid var(--hairline)', zIndex: 2 }}>
      {sortKey ? (
        <button onClick={() => setSort(s => ({ key: sortKey, dir: s.key === sortKey && s.dir === 'desc' ? 'asc' : 'desc' }))}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 4, background: 'none', border: 'none', color: 'inherit', font: 'inherit', textTransform: 'inherit', letterSpacing: 'inherit', flexDirection: align === 'right' ? 'row-reverse' : 'row' }}>
          {label}
          <span style={{ opacity: sort.key === sortKey ? 1 : 0.25, transform: sort.key === sortKey && sort.dir === 'asc' ? 'rotate(180deg)' : 'none', display: 'inline-flex' }}><Icon name="chevDown" size={12} /></span>
        </button>
      ) : label}
    </th>
  );

  return (
    <div style={{ animation: 'om-fade-in .25s ease', display: 'flex', flexDirection: 'column', height: '100%' }}>
      <SectionHeader title="Documentos" subtitle={filtered.length + ' de ' + DB.documents.length + ' documentos'}
        action={
          <div style={{ display: 'flex', gap: 10 }}>
            <Segmented value={density} onChange={setDensity} size="sm" options={[
              { value: 'comfortable', icon: 'linhas', label: '', title: 'Densidade confortável' },
              { value: 'compact', icon: 'linhasComp', label: '', title: 'Densidade compacta' },
            ]} />
            <Button variant="default" icon="download" onClick={() => {
              const b = window.OContabilBridge;
              if (!b || !b.available) { toast('Exportação indisponível'); return; }
              b.call('exports.run', { format: 'csv', clienteId: fCliente === 'todos' ? 0 : fCliente, tipo: fTipo, status: fStatus === 'aprovado' ? 'aprovado' : 'todos' }).then(r => {
                if (r && r.ok && r.data && r.data.canceled) return;
                if (r && r.ok && r.data) toast((r.data.count || 0) + ' documento(s) exportado(s) para CSV');
                else toast((r && r.error) || 'Falha na exportação');
              });
            }}>Exportar</Button>
            <Button variant="default" icon="pasta" onClick={() => setOrgOpen(true)}>Organizar</Button>
            <Button variant="primary" icon="upload" onClick={() => {
              window.OContabilBridge.call('documents.upload', { clienteId: fCliente === 'todos' ? 0 : fCliente }).then(function (r) {
                if (r && r.ok && r.data && r.data.canceled) return;
                if (r && r.ok && r.data) {
                  toast((r.data.enqueued || 0) + ' documento(s) enviado(s) para extração');
                  if (window.__refreshData) { window.__refreshData(); setTimeout(window.__refreshData, 2500); setTimeout(window.__refreshData, 6000); }
                } else { toast((r && r.error) || 'Falha ao importar'); }
              });
            }}>Importar</Button>
          </div>
        } />

      {/* Barra de filtros */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, alignItems: 'center', flexWrap: 'wrap' }}>
        <SearchInput value={search} onChange={setSearch} placeholder="Buscar nº, cliente, chave de acesso…" width={300} />
        <FilterChip label="Cliente" value={fCliente} options={clienteOpts} onChange={setFCliente} icon="clientes" />
        <FilterChip label="Tipo" value={fTipo} options={tipoOpts} onChange={setFTipo} icon="arquivo" />
        <FilterChip label="Status" value={fStatus} options={statusOpts} onChange={setFStatus} icon="ponto" />
        <FilterChip label="Confiança" value={fConf} options={confOpts} onChange={setFConf} icon="escudo" />
        <FilterChip label="Valor" value={fValor} options={valorOpts} onChange={setFValor} icon="cifra" />
        <FilterChip label="Período" value={fPeriodo} options={periodoOpts} onChange={setFPeriodo} icon="calendario" />
        {activeFilters > 0 && (
          <button onClick={clearAll} style={{ display: 'inline-flex', alignItems: 'center', gap: 5, height: 34, padding: '0 10px', fontSize: 12.5, background: 'transparent', border: 'none', color: 'var(--text-2)', borderRadius: 6 }}>
            <Icon name="x" size={13} /> Limpar ({activeFilters})
          </button>
        )}
      </div>

      {/* Barra de seleção em lote */}
      {sel.size > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '9px 14px', marginBottom: 12, background: 'var(--accent-weak)', border: '1px solid var(--accent-weak-2)', borderRadius: 'var(--r-md)', animation: 'om-pop-in .15s ease' }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent-ink)' }}>{sel.size} selecionados</span>
          <div style={{ width: 1, height: 18, background: 'var(--accent-weak-2)' }} />
          <Button variant="ghost" size="sm" icon="check" onClick={() => {
            const b = window.OContabilBridge; const ids = Array.from(sel);
            if (b && b.available) Promise.all(ids.map(id => b.call('documents.validate', { id }))).then(() => { if (window.__refreshData) window.__refreshData(); });
            toast('✓ ' + ids.length + ' documento(s) aprovado(s)'); setSel(new Set());
          }}>Aprovar</Button>
          <Button variant="ghost" size="sm" icon="x" onClick={() => {
            const b = window.OContabilBridge; const ids = Array.from(sel);
            if (b && b.available) Promise.all(ids.map(id => b.call('documents.reject', { id, reason: 'Rejeição em lote' }))).then(() => { if (window.__refreshData) window.__refreshData(); });
            toast(ids.length + ' documento(s) rejeitado(s)'); setSel(new Set());
          }}>Rejeitar</Button>
          <Button variant="ghost" size="sm" icon="download" onClick={() => {
            const b = window.OContabilBridge; const ids = Array.from(sel);
            if (!b || !b.available) { toast('Exportação indisponível'); return; }
            b.call('exports.run', { format: 'csv', ids }).then(r => {
              if (r && r.ok && r.data && r.data.canceled) return;
              if (r && r.ok && r.data) toast((r.data.count || 0) + ' documento(s) exportado(s) para CSV');
              else toast((r && r.error) || 'Falha na exportação');
            });
            setSel(new Set());
          }}>Exportar</Button>
          <div style={{ flex: 1 }} />
          <button onClick={() => setSel(new Set())} style={{ background: 'none', border: 'none', color: 'var(--accent-ink)', fontSize: 12.5, fontWeight: 500 }}>Limpar seleção</button>
        </div>
      )}

      {/* Tabela */}
      <Panel style={{ padding: 0, overflow: 'hidden', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div style={{ overflow: 'auto', flex: 1 }}>
          {filtered.length === 0 ? (
            <EmptyState icon="busca" title="Nenhum documento encontrado" desc="Ajuste os filtros ou o termo de busca para ver resultados."
              action={<Button variant="default" size="sm" icon="x" onClick={clearAll}>Limpar filtros</Button>} />
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ width: 42, padding: '0 0 0 16px', position: 'sticky', top: 0, background: 'var(--surface-2)', borderBottom: '1px solid var(--hairline)', height: 38, zIndex: 2 }}>
                    <input type="checkbox" checked={allSel} onChange={toggleAll} style={{ width: 15, height: 15, accentColor: 'var(--accent)' }} />
                  </th>
                  <Th label="Tipo / Nº" sortKey="cliente" width={210} />
                  <Th label="Cliente" width={180} />
                  <Th label="Chave de acesso" width={180} />
                  <Th label="Valor" sortKey="valor" align="right" width={120} />
                  <Th label="Data" sortKey="data" width={92} />
                  <Th label="Status" width={120} />
                  <Th label="Confiança" sortKey="confianca" width={150} />
                  <th style={{ width: 44, position: 'sticky', top: 0, background: 'var(--surface-2)', borderBottom: '1px solid var(--hairline)' }}></th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((d, i) => {
                  const isSel = sel.has(d.id);
                  return (
                    <tr key={d.id} onClick={() => openDoc(d)} style={{
                      height: 'var(--row-h)', cursor: 'pointer',
                      background: isSel ? 'var(--accent-weak)' : (i % 2 ? 'var(--zebra)' : 'transparent'),
                      borderBottom: '1px solid var(--hairline-2)', transition: 'background .1s',
                    }}
                      onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'var(--subtle)'; }}
                      onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = i % 2 ? 'var(--zebra)' : 'transparent'; }}>
                      <td style={{ padding: '0 0 0 16px' }} onClick={e => { e.stopPropagation(); toggleSel(d.id); }}>
                        <input type="checkbox" checked={isSel} onChange={() => {}} style={{ width: 15, height: 15, accentColor: 'var(--accent)' }} />
                      </td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <span style={{ color: 'var(--text-3)', display: 'inline-flex', flexShrink: 0 }}><Icon name={window.DOC_ICON[d.tipo] || 'arquivo'} size={17} /></span>
                          <div style={{ minWidth: 0 }}>
                            <div style={{ fontWeight: 500, color: 'var(--text)' }}>{d.tipo} <span className="mono">{d.numero}</span></div>
                            <div style={{ fontSize: 11, color: 'var(--text-3)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 160 }}>{d.emitente}</div>
                          </div>
                        </div>
                      </td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)', color: 'var(--text-2)' }}>{d.cliente}</td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)' }}>
                        <span className="mono" style={{ fontSize: 11.5, color: 'var(--text-3)' }}>…{d.chave.slice(-12).replace(/(.{4})/g, '$1 ').trim()}</span>
                      </td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)', textAlign: 'right' }}>
                        <span className="mono" style={{ fontWeight: 500, color: 'var(--text)' }}>R$ {DB.brl(d.valor)}</span>
                      </td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)' }}>
                        <span className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>{d.dataStr}</span>
                        {d.diasParado != null && <div className="mono" style={{ fontSize: 10, color: 'var(--st-review)', fontWeight: 500 }}>{d.diasParado}d parado</div>}
                      </td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)' }}><StatusBadge status={d.status} size="sm" /></td>
                      <td style={{ padding: 'var(--row-pad-y) var(--cell-px)' }}><ConfidenceBar value={d.confianca} width={70} compact={density === 'compact'} /></td>
                      <td style={{ padding: '0 8px 0 0', textAlign: 'right' }}>
                        <span style={{ color: 'var(--text-3)', display: 'inline-flex' }}><Icon name="chevRight" size={16} /></span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderTop: '1px solid var(--hairline)', background: 'var(--surface-2)' }}>
          <span style={{ fontSize: 12, color: 'var(--text-2)' }}>Mostrando <span className="mono" style={{ color: 'var(--text)' }}>{filtered.length}</span> documentos</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Button variant="ghost" size="sm" icon="chevLeft" disabled />
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-2)', padding: '0 8px' }}>1 / 1</span>
            <Button variant="ghost" size="sm" icon="chevRight" disabled />
          </div>
        </div>
      </Panel>
      <OrganizeModal open={orgOpen} onClose={() => setOrgOpen(false)} clienteId={fCliente === 'todos' ? 0 : fCliente} toast={toast} />
    </div>
  );
}

/* ============================================================
   Organização — estrutura de pastas determinística (sem IA)
   Empresa / Ano / Mês / Tipo · cópia local, originais preservados
   ============================================================ */
function OrganizeModal({ open, onClose, clienteId, toast }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);

  useEffect(() => {
    if (!open) { setPlan(null); setReport(null); setRunning(false); return; }
    setLoading(true); setReport(null);
    window.OContabilBridge.call('organize.plan', { clienteId: clienteId || 0 }).then(function (r) {
      setLoading(false);
      if (r && r.ok) setPlan(r.data); else toast((r && r.error) || 'Falha ao planejar organização', 'error');
    });
  }, [open, clienteId]);

  const run = () => {
    setRunning(true);
    window.OContabilBridge.call('organize.run', { clienteId: clienteId || 0 }).then(function (r) {
      setRunning(false);
      if (r && r.ok && r.data && r.data.canceled) return;
      if (r && r.ok && r.data) { setReport(r.data); toast((r.data.Organizados || 0) + ' arquivo(s) organizado(s)'); }
      else toast((r && r.error) || 'Falha ao organizar', 'error');
    });
  };

  return (
    <Modal open={open} onClose={onClose} width={620}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 11, padding: '18px 22px', borderBottom: '1px solid var(--hairline)' }}>
        <span style={{ color: 'var(--accent)' }}><Icon name="pasta" size={20} /></span>
        <div style={{ flex: 1 }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>Organizar arquivos</h2>
          <p style={{ margin: '2px 0 0', fontSize: 12.5, color: 'var(--text-2)' }}>Estrutura por <span className="mono">Empresa / Ano / Mês / Tipo</span> — cópia local, originais preservados.</p>
        </div>
        <button onClick={onClose} style={{ ...miniBtn, width: 30, height: 30 }}><Icon name="x" size={15} /></button>
      </div>

      <div style={{ padding: 22 }}>
        {loading && (
          <div style={{ padding: '28px 0', textAlign: 'center', color: 'var(--text-2)', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 9 }}>
            <Icon name="spinner" size={17} style={{ animation: 'om-spin .8s linear infinite' }} /> Calculando estrutura…
          </div>
        )}

        {!loading && report && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 14px', background: 'var(--st-approved-bg)', border: '1px solid ' + 'var(--st-approved)' + '35', borderRadius: 'var(--r-md)' }}>
              <span style={{ color: 'var(--st-approved)' }}><Icon name="check" size={18} /></span>
              <div style={{ fontSize: 13, color: 'var(--text)' }}><b>{report.Organizados}</b> organizado(s) · {report.Ignorados} já existentes · {report.Erros} erro(s)</div>
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-2)' }}>Pasta: <span className="mono" style={{ wordBreak: 'break-all', color: 'var(--text)' }}>{report.Raiz}</span></div>
            {report.Mensagens && report.Mensagens.length > 0 && (
              <Panel style={{ padding: 12, maxHeight: 140, overflow: 'auto' }}>
                {report.Mensagens.map((m, i) => <div key={i} style={{ fontSize: 11.5, color: 'var(--st-rejected)', fontFamily: 'var(--font-mono)' }}>{m}</div>)}
              </Panel>
            )}
            <Button variant="default" full onClick={onClose}>Fechar</Button>
          </div>
        )}

        {!loading && !report && plan && (
          plan.Total === 0
            ? <EmptyState icon="pasta" title="Nada para organizar" desc="Não há documentos com arquivo disponível para o filtro atual." />
            : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div style={{ display: 'flex', gap: 10 }}>
                  {[['Documentos', plan.Total], ['Com arquivo', plan.ComArquivo], ['Sem arquivo', plan.SemArquivo]].map(([l, v]) => (
                    <Panel key={l} style={{ flex: 1, padding: '12px 14px' }}>
                      <div style={{ fontSize: 11.5, color: 'var(--text-2)' }}>{l}</div>
                      <div className="mono" style={{ fontSize: 20, fontWeight: 600, color: 'var(--text)' }}>{v}</div>
                    </Panel>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>Pré-visualização da estrutura</div>
                  <Panel style={{ padding: 12, maxHeight: 240, overflow: 'auto' }}>
                    {plan.Arvore.map((emp, i) => <TreeNode key={i} node={emp} level={0} />)}
                  </Panel>
                </div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <Button variant="ghost" onClick={onClose}>Cancelar</Button>
                  <div style={{ flex: 1 }} />
                  <Button variant="primary" icon={running ? null : 'pasta'} disabled={running || plan.ComArquivo === 0} onClick={run}>
                    {running ? 'Organizando…' : 'Escolher pasta e organizar ' + plan.ComArquivo}
                  </Button>
                </div>
                <p style={{ margin: 0, fontSize: 10.5, color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                  <Icon name="escudo" size={12} /> Arquivos são copiados (originais preservados) — nada sai desta máquina.
                </p>
              </div>
            )
        )}
      </div>
    </Modal>
  );
}

function TreeNode({ node, level }) {
  const [open, setOpen] = useState(level < 1);
  const hasChildren = node.Filhos && node.Filhos.length > 0;
  return (
    <div style={{ marginLeft: level * 14 }}>
      <button onClick={() => hasChildren && setOpen(o => !o)} style={{
        display: 'flex', alignItems: 'center', gap: 7, width: '100%', padding: '4px', background: 'transparent',
        border: 'none', color: 'var(--text)', textAlign: 'left', fontSize: 12.5, cursor: hasChildren ? 'pointer' : 'default',
      }}>
        {hasChildren
          ? <span style={{ color: 'var(--text-3)', transform: open ? 'rotate(90deg)' : 'none', display: 'inline-flex', transition: 'transform .12s' }}><Icon name="chevRight" size={12} /></span>
          : <span style={{ width: 12, display: 'inline-block' }} />}
        <span style={{ color: level === 0 ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex' }}>
          <Icon name={level === 0 ? 'clientes' : level >= 3 ? 'arquivo' : 'pasta'} size={14} />
        </span>
        <span style={{ flex: 1, fontWeight: level === 0 ? 600 : 400, color: level === 0 ? 'var(--text)' : 'var(--text-2)' }}>{node.Nome}</span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>{node.Total}</span>
      </button>
      {open && hasChildren && node.Filhos.map((c, i) => <TreeNode key={i} node={c} level={level + 1} />)}
    </div>
  );
}

window.DocumentsScreen = DocumentsScreen;
