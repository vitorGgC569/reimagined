/* ============================================================
   OContabil — Clientes (lista + cadastro/edição)
   ============================================================ */

function ClientsScreen({ toast, nav }) {
  const DB = window.DB;
  const [search, setSearch] = useState('');
  const [editOpen, setEditOpen] = useState(false);
  const [editing, setEditing] = useState(null);

  const rows = DB.clients.filter(c =>
    !search || c.nome.toLowerCase().includes(search.toLowerCase()) || c.cnpj.includes(search) || c.fantasia.toLowerCase().includes(search.toLowerCase())
  );

  const openEdit = (c) => { setEditing(c); setEditOpen(true); };
  const openNew = () => { setEditing(null); setEditOpen(true); };

  return (
    <div style={{ animation: 'om-fade-in .25s ease' }}>
      <SectionHeader title="Clientes" subtitle={DB.clients.length + ' empresas cadastradas'} icon="clientes"
        action={<Button variant="primary" icon="mais" onClick={openNew}>Novo cliente</Button>} />

      <div style={{ marginBottom: 14 }}>
        <SearchInput value={search} onChange={setSearch} placeholder="Buscar razão social ou CNPJ…" width={320} />
      </div>

      <Panel style={{ padding: 0, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: 'var(--surface-2)' }}>
              {['Razão social', 'CNPJ', 'UF', 'Regime', 'Schemas', 'Volume', 'Status', ''].map((h, i) => (
                <th key={i} style={{ padding: '11px 14px', textAlign: i === 5 ? 'right' : 'left', fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: '1px solid var(--hairline)', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((c, i) => (
              <tr key={c.id} style={{ borderBottom: '1px solid var(--hairline-2)', background: i % 2 ? 'var(--zebra)' : 'transparent', cursor: 'pointer' }}
                onClick={() => openEdit(c)}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--subtle)'} onMouseLeave={e => e.currentTarget.style.background = i % 2 ? 'var(--zebra)' : 'transparent'}>
                <td style={{ padding: '12px 14px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
                    <Avatar nome={c.fantasia} size={32} />
                    <div>
                      <div style={{ fontWeight: 500, color: 'var(--text)' }}>{c.fantasia}</div>
                      <div style={{ fontSize: 11.5, color: 'var(--text-3)' }}>{c.nome}</div>
                    </div>
                  </div>
                </td>
                <td style={{ padding: '12px 14px' }}><span className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>{c.cnpj}</span></td>
                <td style={{ padding: '12px 14px', color: 'var(--text-2)' }}>{c.uf}</td>
                <td style={{ padding: '12px 14px', color: 'var(--text-2)', fontSize: 12.5 }}>{c.regime}</td>
                <td style={{ padding: '12px 14px' }}>
                  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                    {c.schemas.slice(0, 3).map(s => <span key={s} style={{ fontSize: 10.5, padding: '2px 7px', borderRadius: 4, background: 'var(--subtle)', color: 'var(--text-2)', fontWeight: 500 }}>{s}</span>)}
                    {c.schemas.length > 3 && <span style={{ fontSize: 10.5, color: 'var(--text-3)', padding: '2px 4px' }}>+{c.schemas.length - 3}</span>}
                  </div>
                </td>
                <td style={{ padding: '12px 14px', textAlign: 'right' }}><span className="mono" style={{ fontWeight: 500 }}>{c.volume}</span><span style={{ fontSize: 11, color: 'var(--text-3)' }}>/mês</span></td>
                <td style={{ padding: '12px 14px' }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: c.ativo ? 'var(--st-approved)' : 'var(--text-3)' }}>
                    <span style={{ width: 7, height: 7, borderRadius: 999, background: c.ativo ? 'var(--st-approved)' : 'var(--text-3)' }} />{c.ativo ? 'Ativo' : 'Inativo'}
                  </span>
                </td>
                <td style={{ padding: '12px 14px', textAlign: 'right' }}><span style={{ color: 'var(--text-3)' }}><Icon name="chevRight" size={16} /></span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>

      <ClientEditModal open={editOpen} onClose={() => setEditOpen(false)} client={editing} toast={toast} />
    </div>
  );
}

function ClientEditModal({ open, onClose, client, toast }) {
  const DB = window.DB;
  const isNew = !client;
  const [form, setForm] = useState({ nome: '', fantasia: '', cnpj: '', uf: 'SP', regime: 'Simples Nacional' });
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (open) { setErr(''); setBusy(false); setForm(client ? { nome: client.nome, fantasia: client.fantasia, cnpj: client.cnpj, uf: client.uf, regime: client.regime } : { nome: '', fantasia: '', cnpj: '', uf: 'SP', regime: 'Simples Nacional' }); }
  }, [open, client]);

  // Persistência real via ponte C# (clients.create/update). Fora do WebView2
  // (preview no navegador) mantém o comportamento antigo de protótipo.
  const save = () => {
    const b = window.OContabilBridge;
    if (!b || !b.available) { onClose(); toast(isNew ? 'Cliente cadastrado (protótipo)' : 'Cliente atualizado (protótipo)'); return; }
    setErr(''); setBusy(true);
    const call = isNew
      ? b.call('clients.create', { nome: form.nome, cnpj: form.cnpj, regime: form.regime })
      : b.call('clients.update', { id: client.id, nome: form.nome, regime: form.regime });
    call.then(function (r) {
      setBusy(false);
      if (r && r.ok) { onClose(); toast(isNew ? 'Cliente cadastrado' : 'Cliente atualizado'); if (window.__refreshData) window.__refreshData(); }
      else setErr((r && r.error) || 'Não foi possível salvar o cliente.');
    });
  };

  const deactivate = () => {
    const b = window.OContabilBridge;
    if (!b || !b.available) { onClose(); toast('Cliente desativado (protótipo)'); return; }
    setErr(''); setBusy(true);
    b.call('clients.update', { id: client.id, ativo: !client.ativo }).then(function (r) {
      setBusy(false);
      if (r && r.ok) { onClose(); toast(client.ativo ? 'Cliente desativado' : 'Cliente reativado'); if (window.__refreshData) window.__refreshData(); }
      else setErr((r && r.error) || 'Não foi possível alterar o status.');
    });
  };

  const fld = { width: '100%', height: 40, padding: '0 12px', fontSize: 13.5, background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)', outline: 'none', fontFamily: 'var(--font-sans)' };
  const lbl = { display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--text-2)', marginBottom: 7 };
  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  return (
    <Modal open={open} onClose={onClose} width={560}>
      <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--hairline)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h2 style={{ margin: 0, fontSize: 16.5, fontWeight: 600 }}>{isNew ? 'Novo cliente' : 'Editar cliente'}</h2>
        <button onClick={onClose} style={miniBtn}><Icon name="x" size={16} /></button>
      </div>
      <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div>
          <label style={lbl}>Razão social</label>
          <input style={fld} value={form.nome} onChange={e => set('nome', e.target.value)} placeholder="Empresa Exemplo Ltda" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 14 }}>
          <div>
            <label style={lbl}>Nome fantasia</label>
            <input style={fld} value={form.fantasia} onChange={e => set('fantasia', e.target.value)} />
          </div>
          <div>
            <label style={lbl}>CNPJ</label>
            <input style={{ ...fld, fontFamily: 'var(--font-mono)' }} value={form.cnpj} onChange={e => set('cnpj', e.target.value)} placeholder="00.000.000/0000-00" />
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 14 }}>
          <div>
            <label style={lbl}>UF</label>
            <Select value={form.uf} onChange={v => set('uf', v)} width="100%" options={['SP', 'RJ', 'MG', 'RS', 'PR', 'GO', 'BA', 'SC'].map(u => ({ value: u, label: u }))} />
          </div>
          <div>
            <label style={lbl}>Regime tributário</label>
            <Select value={form.regime} onChange={v => set('regime', v)} width="100%" options={['Simples Nacional', 'Lucro Presumido', 'Lucro Real'].map(r => ({ value: r, label: r }))} />
          </div>
        </div>
        {!isNew && (
          <div>
            <label style={lbl}>Schemas associados</label>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {client.schemas.map(s => <span key={s} style={{ fontSize: 12, padding: '5px 11px', borderRadius: 999, background: 'var(--accent-weak)', color: 'var(--accent-ink)', fontWeight: 500, border: '1px solid var(--accent-weak-2)' }}>{s}</span>)}
              <button style={{ fontSize: 12, padding: '5px 11px', borderRadius: 999, background: 'transparent', border: '1px dashed var(--hairline)', color: 'var(--text-2)', display: 'inline-flex', alignItems: 'center', gap: 5 }}><Icon name="mais" size={12} /> Adicionar</button>
            </div>
          </div>
        )}
      </div>
      {err && <div style={{ margin: '0 22px 14px', fontSize: 12.5, color: 'var(--st-rejected)', background: 'var(--st-rejected-bg)', padding: '9px 12px', borderRadius: 'var(--r-md)', border: '1px solid var(--hairline)' }}>{err}</div>}
      <div style={{ padding: '14px 22px', borderTop: '1px solid var(--hairline)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--surface-2)' }}>
        {!isNew ? <button onClick={deactivate} disabled={busy} style={{ fontSize: 13, color: 'var(--st-rejected)', background: 'none', border: 'none', fontWeight: 500, cursor: 'pointer' }}>{client && client.ativo === false ? 'Reativar cliente' : 'Desativar cliente'}</button> : <span />}
        <div style={{ display: 'flex', gap: 10 }}>
          <Button variant="default" onClick={onClose} disabled={busy}>Cancelar</Button>
          <Button variant="primary" icon="check" onClick={save} disabled={busy}>{busy ? 'Salvando…' : (isNew ? 'Cadastrar' : 'Salvar alterações')}</Button>
        </div>
      </div>
    </Modal>
  );
}

window.ClientsScreen = ClientsScreen;
