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
      if (search) {
        const s = search.toLowerCase();
        if (!(d.numero.includes(s) || d.cliente.toLowerCase().includes(s) || d.emitente.toLowerCase().includes(s) || d.chave.includes(s.replace(/\s/g, '')) || d.tipo.toLowerCase().includes(s))) return false;
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
  }, [search, fCliente, fTipo, fStatus, fConf, sort]);

  const activeFilters = [fCliente, fTipo, fStatus, fConf].filter(v => v !== 'todos').length + (search ? 1 : 0);
  const clearAll = () => { setSearch(''); setFCliente('todos'); setFTipo('todos'); setFStatus('todos'); setFConf('todos'); };

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
            <Button variant="default" icon="download">Exportar</Button>
            <Button variant="primary" icon="upload">Importar</Button>
          </div>
        } />

      {/* Barra de filtros */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, alignItems: 'center', flexWrap: 'wrap' }}>
        <SearchInput value={search} onChange={setSearch} placeholder="Buscar nº, cliente, chave de acesso…" width={300} />
        <FilterChip label="Cliente" value={fCliente} options={clienteOpts} onChange={setFCliente} icon="clientes" />
        <FilterChip label="Tipo" value={fTipo} options={tipoOpts} onChange={setFTipo} icon="arquivo" />
        <FilterChip label="Status" value={fStatus} options={statusOpts} onChange={setFStatus} icon="ponto" />
        <FilterChip label="Confiança" value={fConf} options={confOpts} onChange={setFConf} icon="escudo" />
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
          <Button variant="ghost" size="sm" icon="check" onClick={() => { toast('✓ ' + sel.size + ' documentos aprovados'); setSel(new Set()); }}>Aprovar</Button>
          <Button variant="ghost" size="sm" icon="download" onClick={() => toast('Exportação iniciada para ' + sel.size + ' documentos')}>Exportar</Button>
          <Button variant="ghost" size="sm" icon="x">Rejeitar</Button>
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
    </div>
  );
}

window.DocumentsScreen = DocumentsScreen;
