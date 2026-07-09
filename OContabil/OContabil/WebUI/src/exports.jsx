/* ============================================================
   OContabil — Exportações (SPED, Domínio, Excel, CSV, PDF)
   ============================================================ */

const FORMATOS = [
  { id: 'sped',    nome: 'SPED Fiscal',       ext: '.txt', icon: 'arquivo',  desc: 'Bloco C/D — EFD ICMS/IPID, layout Receita Federal', destaque: true },
  { id: 'dominio', nome: 'Domínio Sistemas',  ext: '.txt', icon: 'planilha', desc: 'Importação direta no Domínio Contábil/Escrita Fiscal', destaque: true },
  { id: 'excel',   nome: 'Excel',             ext: '.xlsx', icon: 'planilha', desc: 'Planilha formatada com abas por tipo de documento' },
  { id: 'csv',     nome: 'CSV (UTF-8 BOM)',   ext: '.csv', icon: 'arquivo',  desc: 'Separado por ponto-e-vírgula, compatível com Excel pt-BR' },
  { id: 'pdf',     nome: 'PDF de conferência', ext: '.pdf', icon: 'olho',    desc: 'Relatório legível para conferência e arquivamento' },
];

function ExportScreen({ toast }) {
  const DB = window.DB;
  const [fmt, setFmt] = useState('sped');
  const [cliente, setCliente] = useState('todos');
  const [tipo, setTipo] = useState('todos');
  const [periodo, setPeriodo] = useState('mes');
  const [status, setStatus] = useState('aprovado');
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [done, setDone] = useState(false);
  const [exportPath, setExportPath] = useState('');

  const sel = useMemo(() => DB.documents.filter(d => {
    if (cliente !== 'todos' && d.clienteId !== cliente) return false;
    if (tipo !== 'todos' && d.tipo !== tipo) return false;
    if (status !== 'todos' && d.status !== status) return false;
    return true;
  }), [cliente, tipo, status]);

  const totalValor = sel.reduce((s, d) => s + d.valor, 0);
  const porTipo = useMemo(() => {
    const m = {};
    sel.forEach(d => { m[d.tipo] = (m[d.tipo] || 0) + 1; });
    return Object.entries(m).sort((a, b) => b[1] - a[1]);
  }, [sel]);

  const fObj = FORMATOS.find(f => f.id === fmt);

  const exportar = () => {
    setRunning(true); setProgress(35); setDone(false);
    window.OContabilBridge.call('exports.run', {
      format: fmt, clienteId: cliente === 'todos' ? 0 : cliente, tipo: tipo, status: status,
    }).then(function (r) {
      setProgress(100); setRunning(false);
      if (r && r.ok && r.data && r.data.canceled) return;
      if (r && r.ok && r.data) {
        setExportPath(r.data.path || ''); setDone(true);
        toast(r.data.count + ' documento(s) exportado(s) — ' + fObj.nome);
      } else {
        toast((r && r.error) || 'Falha na exportação');
      }
    });
  };

  const fieldLabel = { display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--text-2)', marginBottom: 7 };

  return (
    <div style={{ animation: 'om-fade-in .25s ease' }}>
      <SectionHeader title="Exportações" subtitle="Gere arquivos para os sistemas do mercado — tudo nesta máquina" icon="exportacoes" />

      <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 16, alignItems: 'start' }}>
        {/* Coluna config */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Panel style={{ padding: 20 }}>
            <h3 style={{ margin: '0 0 4px', fontSize: 14.5, fontWeight: 600 }}>Formato de exportação</h3>
            <p style={{ margin: '0 0 16px', fontSize: 12.5, color: 'var(--text-2)' }}>Escolha o destino dos documentos</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {FORMATOS.map(f => {
                const active = fmt === f.id;
                return (
                  <button key={f.id} onClick={() => setFmt(f.id)} style={{
                    display: 'flex', gap: 12, padding: '13px 14px', textAlign: 'left', borderRadius: 'var(--r-md)', cursor: 'pointer',
                    background: active ? 'var(--accent-weak)' : 'var(--surface)',
                    border: '1px solid ' + (active ? 'var(--accent)' : 'var(--hairline)'),
                    gridColumn: f.id === 'pdf' ? '1 / -1' : 'auto', transition: 'all .14s', position: 'relative',
                  }}>
                    <span style={{ color: active ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex', flexShrink: 0, marginTop: 1 }}><Icon name={f.icon} size={19} /></span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                        <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>{f.nome}</span>
                        <span className="mono" style={{ fontSize: 10.5, color: 'var(--text-3)' }}>{f.ext}</span>
                        {f.destaque && <span style={{ fontSize: 9.5, fontWeight: 600, color: 'var(--accent)', background: 'var(--surface)', padding: '1px 6px', borderRadius: 999, border: '1px solid var(--accent-weak-2)', letterSpacing: '0.03em' }}>POPULAR</span>}
                      </div>
                      <p style={{ margin: '4px 0 0', fontSize: 11.5, color: 'var(--text-2)', lineHeight: 1.4 }}>{f.desc}</p>
                    </div>
                    <span style={{ width: 17, height: 17, borderRadius: 999, border: '2px solid ' + (active ? 'var(--accent)' : 'var(--hairline)'), display: 'inline-flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      {active && <span style={{ width: 8, height: 8, borderRadius: 999, background: 'var(--accent)' }} />}
                    </span>
                  </button>
                );
              })}
            </div>
          </Panel>

          <Panel style={{ padding: 20 }}>
            <h3 style={{ margin: '0 0 16px', fontSize: 14.5, fontWeight: 600 }}>Filtros do lote</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <div>
                <label style={fieldLabel}>Cliente</label>
                <Select value={cliente} onChange={setCliente} width="100%" options={[{ value: 'todos', label: 'Todos os clientes' }, ...DB.clients.map(c => ({ value: c.id, label: c.fantasia }))]} />
              </div>
              <div>
                <label style={fieldLabel}>Tipo de documento</label>
                <Select value={tipo} onChange={setTipo} width="100%" options={[{ value: 'todos', label: 'Todos os tipos' }, ...DB.docTypes.map(t => ({ value: t, label: t }))]} />
              </div>
              <div>
                <label style={fieldLabel}>Período</label>
                <Select value={periodo} onChange={setPeriodo} width="100%" options={[{ value: 'mes', label: 'Maio/2026' }, { value: 'tri', label: '2º trimestre 2026' }, { value: 'ano', label: 'Ano de 2026' }, { value: 'custom', label: 'Período personalizado…' }]} />
              </div>
              <div>
                <label style={fieldLabel}>Status</label>
                <Select value={status} onChange={setStatus} width="100%" options={[{ value: 'aprovado', label: 'Somente aprovados' }, { value: 'todos', label: 'Todos os status' }]} />
              </div>
            </div>
            {status === 'todos' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '9px 12px', marginTop: 14, background: 'var(--st-review-bg)', border: '1px solid ' + 'var(--st-review)' + '35', borderRadius: 'var(--r-md)' }}>
                <span style={{ color: 'var(--st-review)' }}><Icon name="alerta" size={15} /></span>
                <span style={{ fontSize: 12, color: 'var(--text)' }}>O lote inclui documentos não aprovados. Recomenda-se exportar apenas aprovados para o SPED/Domínio.</span>
              </div>
            )}
          </Panel>
        </div>

        {/* Coluna preview/resumo */}
        <Panel style={{ padding: 0, overflow: 'hidden', position: 'sticky', top: 0 }}>
          <div style={{ padding: '16px 18px', borderBottom: '1px solid var(--hairline)', background: 'var(--surface-2)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
              <span style={{ color: 'var(--accent)' }}><Icon name="olho" size={17} /></span>
              <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Resumo da exportação</h3>
            </div>
          </div>
          <div style={{ padding: 18 }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {[
                ['Formato', fObj.nome + ' ' + fObj.ext],
                ['Documentos', sel.length + ' itens'],
                ['Valor total', 'R$ ' + DB.brl(totalValor)],
                ['Período', periodo === 'tri' ? '2º trimestre 2026' : periodo === 'ano' ? 'Ano 2026' : 'Maio/2026'],
              ].map(([k, v], i) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 0', borderBottom: '1px solid var(--hairline-2)' }}>
                  <span style={{ fontSize: 12.5, color: 'var(--text-2)' }}>{k}</span>
                  <span className={k === 'Valor total' || k === 'Documentos' ? 'mono' : ''} style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>{v}</span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 16 }}>
              <span style={{ fontSize: 11, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Por tipo de documento</span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7, marginTop: 10 }}>
                {porTipo.length === 0 ? <span style={{ fontSize: 12.5, color: 'var(--text-3)' }}>Nenhum documento no filtro atual.</span> :
                  porTipo.map(([t, n]) => (
                    <div key={t} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <span style={{ color: 'var(--text-3)' }}><Icon name={window.DOC_ICON[t] || 'arquivo'} size={15} /></span>
                      <span style={{ fontSize: 12.5, color: 'var(--text-2)', flex: 1 }}>{t}</span>
                      <span className="mono" style={{ fontSize: 12.5, color: 'var(--text)', fontWeight: 500 }}>{n}</span>
                    </div>
                  ))}
              </div>
            </div>

            {running && (
              <div style={{ marginTop: 18 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-2)' }}>Gerando arquivo localmente…</span>
                  <span className="mono" style={{ fontSize: 12, color: 'var(--accent)' }}>{Math.round(progress)}%</span>
                </div>
                <div style={{ height: 6, background: 'var(--subtle)', borderRadius: 999, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: progress + '%', background: 'var(--accent)', borderRadius: 999, transition: 'width .2s' }} />
                </div>
              </div>
            )}

            {done && !running && (
              <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 10, padding: '11px 13px', background: 'var(--st-approved-bg)', border: '1px solid ' + 'var(--st-approved)' + '35', borderRadius: 'var(--r-md)' }}>
                <span style={{ color: 'var(--st-approved)' }}><Icon name="check" size={17} /></span>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text)' }}>Arquivo gerado</div>
                  <div className="mono" style={{ fontSize: 10.5, color: 'var(--text-2)', wordBreak: 'break-all' }}>{exportPath ? exportPath.split(/[\\/]/).pop() : 'export.csv'}</div>
                </div>
                <Button size="sm" variant="default" icon="download">Abrir</Button>
              </div>
            )}

            <Button variant="primary" full size="lg" icon={running ? null : 'download'} onClick={exportar} disabled={running || sel.length === 0} style={{ marginTop: 18 }}>
              {running ? 'Gerando…' : 'Exportar ' + sel.length + ' documentos'}
            </Button>
            <p style={{ margin: '12px 0 0', fontSize: 10.5, color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
              <Icon name="escudo" size={12} /> O arquivo é salvo localmente — nada é enviado para a nuvem
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

window.ExportScreen = ExportScreen;
