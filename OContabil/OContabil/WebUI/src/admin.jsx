/* ============================================================
   OContabil — Usuários + Auditoria
   ============================================================ */

function UsersScreen({ toast }) {
  const DB = window.DB;
  const [tab, setTab] = useState('usuarios');
  const [createOpen, setCreateOpen] = useState(false);
  const [bkPwd, setBkPwd] = useState('');

  const doBackup = (action, okMsg) => {
    if (bkPwd.length < 6) { toast('Senha de backup: mínimo 6 caracteres.', 'error'); return; }
    window.OContabilBridge.call(action, { senha: bkPwd }).then(r => {
      if (r && r.ok && r.data && r.data.canceled) return;       // usuário cancelou o diálogo
      if (r && r.ok) { setBkPwd(''); toast((r.data && r.data.note) || okMsg); }
      else toast((r && r.error) || 'Falha na operação de backup', 'error');
    });
  };

  const papelColor = { 'Administrador': 'var(--accent)', 'Contador': 'var(--st-info)', 'Aux. Contábil': 'var(--text-2)', 'Somente leitura': 'var(--text-3)' };
  const auditIcon = { aprovado: 'check', sistema: 'raio', export: 'download', seguranca: 'cadeado', edicao: 'lapis', config: 'config' };
  const auditColor = { aprovado: 'var(--st-approved)', sistema: 'var(--st-info)', export: 'var(--accent)', seguranca: 'var(--st-rejected)', edicao: 'var(--st-review)', config: 'var(--text-2)' };

  return (
    <div style={{ animation: 'om-fade-in .25s ease' }}>
      <SectionHeader title="Usuários e segurança" subtitle="Gestão de acesso, permissões e trilha de auditoria" icon="usuarios"
        action={<Button variant="primary" icon="mais" onClick={() => setCreateOpen(true)}>Convidar usuário</Button>} />

      {/* Indicadores de segurança */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 16 }}>
        {[
          { l: 'Usuários ativos', v: DB.users.filter(u => u.status === 'ativo').length, ic: 'usuarios', c: 'var(--st-approved)' },
          { l: 'Com MFA ativo', v: DB.users.filter(u => u.mfa).length + '/' + DB.users.length, ic: 'escudo', c: 'var(--accent)' },
          { l: 'Contas bloqueadas', v: DB.users.filter(u => u.status === 'bloqueado').length, ic: 'cadeado', c: 'var(--st-rejected)' },
          { l: 'Eventos de auditoria (7d)', v: 142, ic: 'relogio', c: 'var(--st-info)' },
        ].map(k => (
          <Panel key={k.l} style={{ padding: '15px 17px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <span style={{ fontSize: 12, color: 'var(--text-2)', fontWeight: 500 }}>{k.l}</span>
              <span style={{ color: k.c }}><Icon name={k.ic} size={16} /></span>
            </div>
            <span className="mono" style={{ fontSize: 24, fontWeight: 600, color: 'var(--text)' }}>{k.v}</span>
          </Panel>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 2, marginBottom: 14, borderBottom: '1px solid var(--hairline)' }}>
        {[['usuarios', 'Usuários'], ['auditoria', 'Trilha de auditoria'], ['backup', 'Backup e recuperação']].map(([k, l]) => (
          <button key={k} onClick={() => setTab(k)} style={{ padding: '9px 14px', fontSize: 13.5, fontWeight: 500, background: 'transparent', border: 'none', borderBottom: '2px solid ' + (tab === k ? 'var(--accent)' : 'transparent'), color: tab === k ? 'var(--text)' : 'var(--text-2)', marginBottom: -1 }}>{l}</button>
        ))}
      </div>

      {tab === 'usuarios' && (
        <Panel style={{ padding: 0, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead><tr style={{ background: 'var(--surface-2)' }}>
              {['Usuário', 'Papel', 'Último login', 'MFA', 'Tentativas', 'Status'].map((h, i) => (
                <th key={i} style={{ padding: '11px 16px', textAlign: 'left', fontSize: 11.5, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: '1px solid var(--hairline)' }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {DB.users.map((u, i) => (
                <tr key={u.id} style={{ borderBottom: '1px solid var(--hairline-2)', background: i % 2 ? 'var(--zebra)' : 'transparent' }}>
                  <td style={{ padding: '12px 16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
                      <Avatar nome={u.nome} size={32} />
                      <div><div style={{ fontWeight: 500 }}>{u.nome}</div><div className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>{u.email}</div></div>
                    </div>
                  </td>
                  <td style={{ padding: '12px 16px' }}><span style={{ fontSize: 12, fontWeight: 500, color: papelColor[u.papel] }}>{u.papel}</span></td>
                  <td style={{ padding: '12px 16px' }}><span className="mono" style={{ fontSize: 12, color: 'var(--text-2)' }}>{u.ultimoLogin}</span></td>
                  <td style={{ padding: '12px 16px' }}>
                    {u.mfa ? <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 12, color: 'var(--st-approved)' }}><Icon name="escudo" size={13} /> Ativo</span>
                      : <span style={{ fontSize: 12, color: 'var(--text-3)' }}>—</span>}
                  </td>
                  <td style={{ padding: '12px 16px' }}><span className="mono" style={{ fontSize: 12.5, color: u.tentativas >= 3 ? 'var(--st-rejected)' : u.tentativas > 0 ? 'var(--st-review)' : 'var(--text-3)', fontWeight: u.tentativas > 0 ? 600 : 400 }}>{u.tentativas}</span></td>
                  <td style={{ padding: '12px 16px' }}>
                    {u.status === 'ativo' && <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--st-approved)' }}><span style={{ width: 7, height: 7, borderRadius: 999, background: 'var(--st-approved)' }} />Ativo</span>}
                    {u.status === 'bloqueado' && <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--st-rejected)', fontWeight: 500 }}><Icon name="cadeado" size={12} />Bloqueado</span>}
                    {u.status === 'convidado' && <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-3)' }}><Icon name="relogio" size={12} />Convidado</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      {tab === 'auditoria' && (
        <Panel style={{ padding: 22 }}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {DB.auditoria.map((a, i) => (
              <div key={i} style={{ display: 'flex', gap: 13, position: 'relative' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                  <span style={{ width: 30, height: 30, borderRadius: 999, background: auditColor[a.tipo] + '18', color: auditColor[a.tipo], display: 'inline-flex', alignItems: 'center', justifyContent: 'center', border: '1px solid ' + auditColor[a.tipo] + '30' }}><Icon name={auditIcon[a.tipo]} size={15} /></span>
                  {i < DB.auditoria.length - 1 && <span style={{ width: 1.5, flex: 1, background: 'var(--hairline)', marginTop: 3, minHeight: 16 }} />}
                </div>
                <div style={{ paddingBottom: i < DB.auditoria.length - 1 ? 18 : 0, flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                    <span style={{ fontSize: 13, color: 'var(--text)' }}>{a.acao}</span>
                    <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)', whiteSpace: 'nowrap' }}>{a.quando}</span>
                  </div>
                  <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 4, display: 'flex', gap: 12 }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><Icon name="usuarios" size={12} /> {a.quem}</span>
                    <span className="mono" style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><Icon name="escudo" size={11} /> {a.ip}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      )}
      {tab === 'backup' && (
        <Panel style={{ padding: 22, maxWidth: 640 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{ color: 'var(--accent)' }}><Icon name="cadeado" size={18} /></span>
            <span style={{ fontSize: 15, fontWeight: 600 }}>Backup cifrado por senha</span>
          </div>
          <p style={{ fontSize: 12.5, color: 'var(--text-2)', lineHeight: 1.55, marginBottom: 16 }}>
            O banco é cifrado em repouso com uma chave protegida pela sua conta do Windows. Se você
            reinstalar o Windows, trocar de conta ou o perfil corromper, essa chave se perde e o banco
            fica irrecuperável. Gere um backup cifrado por senha e guarde o arquivo
            <span className="mono"> .ocbak</span> + a senha em local seguro — é o seu plano de recuperação.
          </p>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--text-2)', marginBottom: 6 }}>Senha do backup (mín. 6 caracteres)</label>
          <input type="password" value={bkPwd} onChange={e => setBkPwd(e.target.value)} placeholder="••••••••"
            style={{ width: '100%', padding: '9px 12px', fontSize: 13, background: 'var(--surface-2)', border: '1px solid var(--hairline)', borderRadius: 8, color: 'var(--text)', marginBottom: 16, boxSizing: 'border-box' }} />
          <div style={{ display: 'flex', gap: 10 }}>
            <Button variant="primary" icon="download" onClick={() => doBackup('backup.export', 'Backup exportado.')}>Exportar backup</Button>
            <Button variant="default" icon="upload" onClick={() => doBackup('backup.restore', 'Restauração preparada — reinicie o app.')}>Restaurar backup…</Button>
          </div>
          <p style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 14 }}>
            A restauração é validada e aplicada ao reabrir o OContabil. Operação restrita a administradores.
          </p>
        </Panel>
      )}

      <UserCreateModal open={createOpen} onClose={() => setCreateOpen(false)} toast={toast} />
    </div>
  );
}

/* ============================================================
   Schemas (campos customizados por cliente/tipo)
   ============================================================ */
function SchemasScreen({ toast }) {
  const DB = window.DB;
  const schemaList = (DB.schemas && DB.schemas.length ? DB.schemas : []).map(function (s) {
    return {
      id: s.id, nome: s.nome, tipo: s.tipo, sistema: s.sistema,
      clientes: s.clienteId ? 1 : 0, campos: (s.campos || []).length,
      fields: s.campos || [], atualizado: s.sistema ? 'sistema (global)' : 'cliente',
    };
  });
  const [sel, setSel] = useState(schemaList[0] || null);
  const [schemaModal, setSchemaModal] = useState(null);
  if (!sel) return (
    <div style={{ padding: 28, color: 'var(--text-2)', animation: 'om-fade-in .25s ease' }}>
      Nenhum schema de extração cadastrado.
      <div style={{ marginTop: 14 }}><Button variant="primary" icon="mais" onClick={() => setSchemaModal({ mode: 'new' })}>Novo schema</Button></div>
      <SchemaEditModal state={schemaModal} onClose={() => setSchemaModal(null)} toast={toast} />
    </div>
  );
  const campos = (sel.fields || []).map(function (f) {
    return { campo: f.nome, tipoVal: (f.tipo === 'str' || !f.tipo ? 'texto' : f.tipo), desc: f.desc || '' };
  });

  return (
    <div style={{ animation: 'om-fade-in .25s ease' }}>
      <SectionHeader title="Schemas de extração" subtitle="Defina quais campos extrair por tipo de documento e cliente" icon="schemas"
        action={<Button variant="primary" icon="mais" onClick={() => setSchemaModal({ mode: 'new' })}>Novo schema</Button>} />

      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 16, alignItems: 'start' }}>
        <Panel style={{ padding: 8 }}>
          {schemaList.map(s => (
            <button key={s.id} onClick={() => setSel(s)} style={{
              display: 'flex', alignItems: 'center', gap: 11, width: '100%', padding: '11px 12px', textAlign: 'left',
              background: sel.id === s.id ? 'var(--accent-weak)' : 'transparent', border: 'none', borderRadius: 'var(--r-md)', marginBottom: 2,
            }}
              onMouseEnter={e => { if (sel.id !== s.id) e.currentTarget.style.background = 'var(--zebra)'; }}
              onMouseLeave={e => { if (sel.id !== s.id) e.currentTarget.style.background = 'transparent'; }}>
              <span style={{ color: sel.id === s.id ? 'var(--accent)' : 'var(--text-3)' }}><Icon name={window.DOC_ICON[s.tipo] || 'arquivo'} size={18} /></span>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: sel.id === s.id ? 'var(--accent-ink)' : 'var(--text)' }}>{s.nome}</div>
                <div style={{ fontSize: 11, color: 'var(--text-3)' }}>{s.campos} campos · {s.clientes} clientes</div>
              </div>
            </button>
          ))}
        </Panel>

        <Panel style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--hairline)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>{sel.nome}</h3>
              <p style={{ margin: '3px 0 0', fontSize: 12, color: 'var(--text-2)' }}>Tipo {sel.tipo} · atualizado em {sel.atualizado}</p>
            </div>
            <Button variant="default" size="sm" icon="mais" disabled={sel.sistema} onClick={() => setSchemaModal({ mode: 'edit', schema: sel })}>Adicionar campo</Button>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead><tr style={{ background: 'var(--surface-2)' }}>
              {['Campo', 'Tipo de dado', 'Obrigatório', 'Validação', ''].map((h, i) => (
                <th key={i} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 11, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: '1px solid var(--hairline)' }}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {campos.map((c, i) => {
                const tipoLabel = { texto: 'Texto', numero: 'Número', cnpj: 'CNPJ', data: 'Data', moeda: 'Moeda (R$)', chave: 'Chave 44díg' }[c.tipoVal];
                const obrig = !['Multa', 'Juros', 'Série'].includes(c.campo);
                return (
                  <tr key={i} style={{ borderBottom: '1px solid var(--hairline-2)', background: i % 2 ? 'var(--zebra)' : 'transparent' }}>
                    <td style={{ padding: '11px 16px', fontWeight: 500 }}>{c.campo}</td>
                    <td style={{ padding: '11px 16px' }}><span className="mono" style={{ fontSize: 11.5, padding: '2px 8px', borderRadius: 4, background: 'var(--subtle)', color: 'var(--text-2)' }}>{tipoLabel}</span></td>
                    <td style={{ padding: '11px 16px' }}>
                      {obrig ? <span style={{ fontSize: 12, color: 'var(--st-rejected)', fontWeight: 500 }}>Obrigatório</span> : <span style={{ fontSize: 12, color: 'var(--text-3)' }}>Opcional</span>}
                    </td>
                    <td style={{ padding: '11px 16px', fontSize: 12, color: 'var(--text-2)' }}>{c.desc || (c.tipoVal === 'cnpj' ? 'Dígito verificador' : c.tipoVal === 'moeda' ? '≥ 0,00' : c.tipoVal === 'data' ? 'dd/mm/aaaa' : '—')}</td>
                    <td style={{ padding: '11px 16px', textAlign: 'right' }}><button style={{ ...miniBtn, width: 28, height: 28 }}><Icon name="lapis" size={13} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      </div>
      <SchemaEditModal state={schemaModal} onClose={() => setSchemaModal(null)} toast={toast} />
    </div>
  );
}

/* ---- estilos de campo (modais admin) ---- */
const admFld = { width: '100%', height: 38, padding: '0 12px', fontSize: 13.5, background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)', outline: 'none', fontFamily: 'var(--font-sans)' };
const admLbl = { display: 'block', fontSize: 12, fontWeight: 500, color: 'var(--text-2)', margin: '0 0 6px' };

/* ---- Criar usuário ---- */
function UserCreateModal({ open, onClose, toast }) {
  const [nome, setNome] = useState(''); const [usuario, setUsuario] = useState('');
  const [email, setEmail] = useState(''); const [papel, setPapel] = useState('Operador');
  const [senha, setSenha] = useState(''); const [busy, setBusy] = useState(false);
  useEffect(() => { if (open) { setNome(''); setUsuario(''); setEmail(''); setPapel('Operador'); setSenha(''); setBusy(false); } }, [open]);
  const salvar = () => {
    if (!nome.trim() || !usuario.trim() || senha.length < 6) { toast('Preencha nome, login e senha (mín. 6).', 'error'); return; }
    setBusy(true);
    window.OContabilBridge.call('users.create', { nome, usuario, email, papel, senha }).then(r => {
      setBusy(false);
      if (r && r.ok) { toast('Usuário criado'); if (window.__refreshData) window.__refreshData(); onClose(); }
      else toast((r && r.error) || 'Falha ao criar usuário', 'error');
    });
  };
  return (
    <Modal open={open} onClose={onClose} width={460}>
      <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--hairline)', display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ color: 'var(--accent)' }}><Icon name="usuarios" size={19} /></span>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Novo usuário</h2>
      </div>
      <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 13 }}>
        <div><label style={admLbl}>Nome completo</label><input style={admFld} value={nome} onChange={e => setNome(e.target.value)} /></div>
        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}><label style={admLbl}>Login</label><input style={admFld} value={usuario} onChange={e => setUsuario(e.target.value)} /></div>
          <div style={{ flex: 1 }}><label style={admLbl}>Senha (mín. 6)</label><input type="password" style={admFld} value={senha} onChange={e => setSenha(e.target.value)} /></div>
        </div>
        <div><label style={admLbl}>E-mail</label><input style={admFld} value={email} onChange={e => setEmail(e.target.value)} /></div>
        <div><label style={admLbl}>Papel</label><Select value={papel} onChange={setPapel} width="100%" options={[{ value: 'Operador', label: 'Operador' }, { value: 'Administrador', label: 'Administrador' }, { value: 'Visualizador', label: 'Visualizador' }]} /></div>
        <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
          <Button variant="ghost" onClick={onClose}>Cancelar</Button>
          <div style={{ flex: 1 }} />
          <Button variant="primary" icon="check" disabled={busy} onClick={salvar}>{busy ? 'Criando…' : 'Criar usuário'}</Button>
        </div>
      </div>
    </Modal>
  );
}

/* ---- Criar / editar schema ---- */
function SchemaEditModal({ state, onClose, toast }) {
  const open = !!state;
  const isNew = !!state && state.mode === 'new';
  const [nome, setNome] = useState(''); const [tipo, setTipo] = useState('');
  const [campos, setCampos] = useState(''); const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!state) return;
    if (state.mode === 'new') { setNome(''); setTipo(''); setCampos(''); }
    else { const s = state.schema; setNome(s.nome); setTipo(s.tipo); setCampos((s.fields || []).map(f => f.nome + '::' + (f.tipo || 'str') + '::' + (f.desc || '')).join('\n')); }
    setBusy(false);
  }, [state]);
  if (!open) return null;
  const salvar = () => {
    const lista = campos.split('\n').map(l => l.trim()).filter(Boolean);
    if (isNew && (!nome.trim() || !tipo.trim())) { toast('Informe nome e tipo.', 'error'); return; }
    setBusy(true);
    const call = isNew
      ? window.OContabilBridge.call('schemas.create', { nome, tipo, campos: lista })
      : window.OContabilBridge.call('schemas.update', { id: state.schema.id, campos: lista });
    call.then(r => {
      setBusy(false);
      if (r && r.ok) { toast(isNew ? 'Schema criado' : 'Schema atualizado'); if (window.__refreshData) window.__refreshData(); onClose(); }
      else toast((r && r.error) || 'Falha ao salvar schema', 'error');
    });
  };
  return (
    <Modal open={open} onClose={onClose} width={520}>
      <div style={{ padding: '18px 22px', borderBottom: '1px solid var(--hairline)', display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ color: 'var(--accent)' }}><Icon name="schemas" size={19} /></span>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{isNew ? 'Novo schema' : 'Editar campos — ' + nome}</h2>
      </div>
      <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 13 }}>
        {isNew && <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}><label style={admLbl}>Nome</label><input style={admFld} value={nome} onChange={e => setNome(e.target.value)} /></div>
          <div style={{ flex: 1 }}><label style={admLbl}>Tipo (ex.: NF-e)</label><input style={admFld} value={tipo} onChange={e => setTipo(e.target.value)} /></div>
        </div>}
        <div>
          <label style={admLbl}>Campos — um por linha (<span className="mono">nome::str::descrição</span>)</label>
          <textarea value={campos} onChange={e => setCampos(e.target.value)} rows={8} style={{ ...admFld, height: 'auto', padding: 10, fontFamily: 'var(--font-mono)', fontSize: 12.5, resize: 'vertical' }} />
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <Button variant="ghost" onClick={onClose}>Cancelar</Button>
          <div style={{ flex: 1 }} />
          <Button variant="primary" icon="check" disabled={busy} onClick={salvar}>{busy ? 'Salvando…' : 'Salvar'}</Button>
        </div>
      </div>
    </Modal>
  );
}

window.UsersScreen = UsersScreen;
window.SchemasScreen = SchemasScreen;
