/* ============================================================
   OContabil — App shell (sidebar, topbar, roteamento, Cmd+K)
   ============================================================ */

const NAV = [
  { id: 'painel', label: 'Painel', icon: 'painel' },
  { id: 'documentos', label: 'Documentos', icon: 'documentos', badge: () => window.DB.documents.filter(d => d.status === 'revisar').length },
  { id: 'clientes', label: 'Clientes', icon: 'clientes' },
  { id: 'schemas', label: 'Schemas', icon: 'schemas' },
  { id: 'exportacoes', label: 'Exportações', icon: 'exportacoes' },
  { id: 'usuarios', label: 'Usuários', icon: 'usuarios' },
  { id: 'config', label: 'Configurações', icon: 'config' },
];

function Sidebar({ route, nav, collapsed, setCollapsed }) {
  const w = collapsed ? 64 : 232;
  return (
    <aside style={{ width: w, flexShrink: 0, background: 'var(--surface)', borderRight: '1px solid var(--hairline)', display: 'flex', flexDirection: 'column', transition: 'width .2s ease' }}>
      <div style={{ height: 60, display: 'flex', alignItems: 'center', padding: collapsed ? '0' : '0 18px', justifyContent: collapsed ? 'center' : 'flex-start', borderBottom: '1px solid var(--hairline)' }}>
        <Logo size={28} showText={!collapsed} />
      </div>

      <nav style={{ flex: 1, padding: 10, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {NAV.map(item => {
          const active = route === item.id;
          const badge = item.badge ? item.badge() : 0;
          return (
            <button key={item.id} onClick={() => nav(item.id)} title={collapsed ? item.label : undefined} style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: collapsed ? '10px 0' : '10px 12px',
              justifyContent: collapsed ? 'center' : 'flex-start', borderRadius: 'var(--r-md)', border: 'none',
              background: active ? 'var(--accent-weak)' : 'transparent', color: active ? 'var(--accent-ink)' : 'var(--text-2)',
              fontSize: 13.5, fontWeight: active ? 600 : 500, position: 'relative', transition: 'background .12s',
            }}
              onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--zebra)'; }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent'; }}>
              {active && <span style={{ position: 'absolute', left: collapsed ? 6 : 0, top: '50%', transform: 'translateY(-50%)', width: 3, height: 18, borderRadius: 999, background: 'var(--accent)' }} />}
              <span style={{ color: active ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex', flexShrink: 0 }}><Icon name={item.icon} size={19} /></span>
              {!collapsed && <span style={{ flex: 1, textAlign: 'left' }}>{item.label}</span>}
              {!collapsed && badge > 0 && <span className="mono" style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--st-review)', background: 'var(--st-review-bg)', padding: '1px 7px', borderRadius: 999 }}>{badge}</span>}
            </button>
          );
        })}
      </nav>

      <div style={{ padding: 10, borderTop: '1px solid var(--hairline)' }}>
        {!collapsed && <div style={{ padding: '0 4px 10px' }}><LocalSeal variant="compact" /></div>}
        <button onClick={() => setCollapsed(c => !c)} style={{ display: 'flex', alignItems: 'center', gap: 11, width: '100%', padding: collapsed ? '9px 0' : '9px 12px', justifyContent: collapsed ? 'center' : 'flex-start', background: 'transparent', border: 'none', borderRadius: 'var(--r-md)', color: 'var(--text-3)', fontSize: 12.5 }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--zebra)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
          <span style={{ display: 'inline-flex', transform: collapsed ? 'none' : 'rotate(180deg)' }}><Icon name="chevRight" size={17} /></span>
          {!collapsed && <span>Recolher menu</span>}
        </button>
      </div>
    </aside>
  );
}

function Topbar({ route, onCommand, onCopilot, theme, setTheme, density, setDensity, onLogout }) {
  const titles = { painel: 'Painel', documentos: 'Documentos', clientes: 'Clientes', schemas: 'Schemas', exportacoes: 'Exportações', usuarios: 'Usuários', config: 'Configurações' };
  const [menuOpen, setMenuOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const h = e => { if (ref.current && !ref.current.contains(e.target)) setMenuOpen(false); };
    document.addEventListener('mousedown', h); return () => document.removeEventListener('mousedown', h);
  }, []);

  return (
    <header style={{ height: 60, flexShrink: 0, background: 'var(--surface)', borderBottom: '1px solid var(--hairline)', display: 'flex', alignItems: 'center', gap: 14, padding: '0 18px' }}>
      <div style={{ fontSize: 12, color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: 7 }}>
        <span>Escritório Razão</span><Icon name="chevRight" size={13} /><span style={{ color: 'var(--text)', fontWeight: 500 }}>{titles[route]}</span>
      </div>

      <div style={{ flex: 1 }} />

      {/* Command bar trigger */}
      <button onClick={onCommand} style={{
        display: 'flex', alignItems: 'center', gap: 9, height: 36, padding: '0 11px 0 12px', minWidth: 240,
        background: 'var(--paper)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text-3)', fontSize: 13,
      }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--text-3)'} onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--hairline)'}>
        <Icon name="busca" size={15} />
        <span style={{ flex: 1, textAlign: 'left' }}>Buscar ou comandar…</span>
        <kbd style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, padding: '2px 6px', borderRadius: 4, background: 'var(--surface)', border: '1px solid var(--hairline)', color: 'var(--text-2)' }}>⌘K</kbd>
      </button>

      {/* Copiloto */}
      <Button variant="default" size="md" icon="copiloto" onClick={onCopilot}>Copiloto</Button>

      <div style={{ width: 1, height: 24, background: 'var(--hairline)' }} />

      {/* Tema */}
      <Tip label={theme === 'light' ? 'Modo escuro' : 'Modo claro'}>
        <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} style={{ ...miniBtn, width: 36, height: 36 }}>
          <Icon name={theme === 'light' ? 'lua' : 'sol'} size={17} />
        </button>
      </Tip>

      {/* Usuário */}
      <div ref={ref} style={{ position: 'relative' }}>
        <button onClick={() => setMenuOpen(o => !o)} style={{ display: 'flex', alignItems: 'center', gap: 8, background: 'transparent', border: 'none', padding: 2, borderRadius: 999 }}>
          <Avatar nome="Ana Beatriz Souza" size={32} />
          <Icon name="chevDown" size={14} style={{ color: 'var(--text-3)' }} />
        </button>
        {menuOpen && (
          <div style={{ position: 'absolute', top: 'calc(100% + 8px)', right: 0, width: 220, background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', boxShadow: 'var(--shadow-2)', padding: 7, zIndex: 60, animation: 'om-pop-in .14s ease' }}>
            <div style={{ padding: '8px 10px 10px', borderBottom: '1px solid var(--hairline-2)', marginBottom: 5 }}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>Ana Beatriz Souza</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--text-3)' }}>Administrador · CAMP-04</div>
            </div>
            {[['usuarios', 'Minha conta'], ['config', 'Configurações'], ['escudo', 'Segurança']].map(([ic, l]) => (
              <button key={l} style={menuItem} onMouseEnter={e => e.currentTarget.style.background = 'var(--zebra)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <Icon name={ic} size={15} /> {l}
              </button>
            ))}
            <div style={{ height: 1, background: 'var(--hairline-2)', margin: '5px 0' }} />
            <button onClick={onLogout} style={{ ...menuItem, color: 'var(--st-rejected)' }} onMouseEnter={e => e.currentTarget.style.background = 'var(--st-rejected-bg)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
              <Icon name="sair" size={15} /> Sair
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
const menuItem = { display: 'flex', alignItems: 'center', gap: 10, width: '100%', padding: '8px 10px', fontSize: 13, background: 'transparent', border: 'none', borderRadius: 5, color: 'var(--text-2)', textAlign: 'left' };

/* ---- Paleta de comandos (Cmd+K) ---- */
function CommandPalette({ open, onClose, nav, openCopilot, setTheme, theme }) {
  const DB = window.DB;
  const [q, setQ] = useState('');
  const inputRef = useRef(null);
  const [idx, setIdx] = useState(0);
  useEffect(() => { if (open) { setQ(''); setIdx(0); setTimeout(() => inputRef.current && inputRef.current.focus(), 50); } }, [open]);

  const commands = useMemo(() => {
    const base = [
      ...NAV.map(n => ({ tipo: 'Navegar', label: n.label, icon: n.icon, run: () => { nav(n.id); onClose(); } })),
      { tipo: 'Ação', label: 'Perguntar ao Copiloto', icon: 'copiloto', run: () => { openCopilot(q); onClose(); } },
      { tipo: 'Ação', label: 'Importar documentos', icon: 'upload', run: () => { nav('documentos'); onClose(); } },
      { tipo: 'Ação', label: 'Nova exportação SPED/Domínio', icon: 'download', run: () => { nav('exportacoes'); onClose(); } },
      { tipo: 'Ação', label: theme === 'light' ? 'Ativar modo escuro' : 'Ativar modo claro', icon: theme === 'light' ? 'lua' : 'sol', run: () => { setTheme(theme === 'light' ? 'dark' : 'light'); onClose(); } },
    ];
    const docs = DB.documents.filter(d => q && (d.numero.includes(q) || d.cliente.toLowerCase().includes(q.toLowerCase()))).slice(0, 4)
      .map(d => ({ tipo: 'Documento', label: d.tipo + ' ' + d.numero + ' · ' + d.cliente, icon: window.DOC_ICON[d.tipo] || 'arquivo', doc: d, run: () => { window.__openDoc(d); onClose(); } }));
    let list = base;
    if (q) list = base.filter(c => c.label.toLowerCase().includes(q.toLowerCase()));
    return [...docs, ...list];
  }, [q, theme]);

  useEffect(() => {
    if (!open) return;
    const onKey = e => {
      if (e.key === 'ArrowDown') { e.preventDefault(); setIdx(i => Math.min(commands.length - 1, i + 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setIdx(i => Math.max(0, i - 1)); }
      else if (e.key === 'Enter') { e.preventDefault(); commands[idx] && commands[idx].run(); }
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, [open, commands, idx]);

  if (!open) return null;
  let lastTipo = null;
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 300, background: 'rgba(22,32,28,0.4)', backdropFilter: 'blur(2px)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: '12vh', animation: 'om-fade-in .12s ease' }}>
      <div onClick={e => e.stopPropagation()} style={{ width: 600, maxWidth: '92vw', background: 'var(--surface)', borderRadius: 'var(--r-lg)', boxShadow: 'var(--shadow-3)', border: '1px solid var(--hairline)', overflow: 'hidden', animation: 'om-pop-in .16s ease' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 11, padding: '14px 16px', borderBottom: '1px solid var(--hairline)' }}>
          <Icon name="busca" size={18} style={{ color: 'var(--text-3)' }} />
          <input ref={inputRef} value={q} onChange={e => { setQ(e.target.value); setIdx(0); }} placeholder="Buscar documentos, navegar ou perguntar ao copiloto…"
            style={{ flex: 1, border: 'none', background: 'transparent', fontSize: 15, color: 'var(--text)', outline: 'none', fontFamily: 'var(--font-sans)' }} />
          <kbd style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '2px 7px', borderRadius: 4, background: 'var(--subtle)', color: 'var(--text-2)' }}>esc</kbd>
        </div>
        <div style={{ maxHeight: 380, overflow: 'auto', padding: 8 }}>
          {commands.length === 0 && <div style={{ padding: '32px 16px', textAlign: 'center', fontSize: 13, color: 'var(--text-3)' }}>Nenhum resultado. Pressione Enter para perguntar ao copiloto.</div>}
          {commands.map((c, i) => {
            const showHeader = c.tipo !== lastTipo; lastTipo = c.tipo;
            return (
              <React.Fragment key={i}>
                {showHeader && <div style={{ fontSize: 10.5, fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '10px 10px 5px' }}>{c.tipo}</div>}
                <button onClick={c.run} onMouseEnter={() => setIdx(i)} style={{
                  display: 'flex', alignItems: 'center', gap: 12, width: '100%', padding: '9px 11px', borderRadius: 'var(--r-md)', border: 'none', textAlign: 'left',
                  background: idx === i ? 'var(--accent-weak)' : 'transparent', color: idx === i ? 'var(--accent-ink)' : 'var(--text)',
                }}>
                  <span style={{ color: idx === i ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex' }}><Icon name={c.icon} size={17} /></span>
                  <span style={{ flex: 1, fontSize: 13.5 }}>{c.label}</span>
                  {idx === i && <Icon name="setaDir" size={15} style={{ color: 'var(--accent)' }} />}
                </button>
              </React.Fragment>
            );
          })}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '9px 16px', borderTop: '1px solid var(--hairline)', background: 'var(--surface-2)', fontSize: 11, color: 'var(--text-3)' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><kbd style={kbdSm}>↑</kbd><kbd style={kbdSm}>↓</kbd> navegar</span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}><kbd style={kbdSm}>↵</kbd> selecionar</span>
          <div style={{ flex: 1 }} />
          <LocalSeal variant="compact" />
        </div>
      </div>
    </div>
  );
}
const kbdSm = { fontFamily: 'var(--font-mono)', fontSize: 10, padding: '1px 5px', borderRadius: 3, background: 'var(--surface)', border: '1px solid var(--hairline)', color: 'var(--text-2)' };

/* ============================================================
   App raiz
   ============================================================ */
function App() {
  const [logged, setLogged] = useState(false);
  const [route, setRoute] = useState('painel');
  const [routeArg, setRouteArg] = useState(null);
  const [theme, setTheme] = useState('light');
  const [density, setDensity] = useState('comfortable');
  const [collapsed, setCollapsed] = useState(false);
  const [doc, setDoc] = useState(null);
  const [cmdOpen, setCmdOpen] = useState(false);
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotFocus, setCopilotFocus] = useState(false);
  const [toast, setToastState] = useState(null);
  const [dataVersion, setDataVersion] = useState(0);

  // Hidrata window.DB com dados reais dos serviços C# (via ponte) após login.
  // Mantém o mock como fallback fora do WebView2. window.__refreshData re-busca
  // tudo (usado após upload/reprocessamento). Bump em dataVersion remonta telas.
  useEffect(() => {
    if (!logged) return;
    var b = window.OContabilBridge;
    if (!b || !b.available) return;
    var alive = true;

    function hydrate() {
      return Promise.all([
        b.call('clients.list'), b.call('documents.list'), b.call('schemas.list'),
        b.call('users.list'), b.call('audit.list'), b.call('dashboard.metrics'),
      ]).then(function (res) {
        if (!alive) return;
        var cr = res[0], dr = res[1], sr = res[2], ur = res[3], ar = res[4], mr = res[5];

        if (cr && cr.ok && Array.isArray(cr.data)) {
          window.DB.clients = cr.data.map(function (c) {
            return {
              id: c.id, nome: c.razaoSocial, fantasia: c.razaoSocial, cnpj: c.cnpj,
              uf: '—', municipio: '—', regime: c.regime || '—', schemas: [],
              volume: c.volume || 0, ativo: c.ativo !== false,
            };
          });
        }
        if (dr && dr.ok && Array.isArray(dr.data)) {
          window.DB.documents = dr.data.map(function (d) {
            return {
              id: d.id, arquivo: d.arquivo, tipo: d.tipo || '—',
              clienteId: d.clienteId, cliente: d.cliente || '—',
              emitente: d.emitente || '', numero: d.numero || String(d.id),
              serie: d.serie || '', chave: d.chave || '',
              valor: d.valor || 0, data: d.data ? new Date(d.data) : new Date(),
              dataStr: d.dataStr || '', dataHora: d.dataHora || '',
              status: d.status || 'pendente',
              confianca: (d.confianca == null ? null : d.confianca),
              diasParado: (d.diasParado == null ? null : d.diasParado),
              revisor: d.revisor || null,
            };
          });
        }
        if (sr && sr.ok && Array.isArray(sr.data)) window.DB.schemas = sr.data;
        if (ur && ur.ok && Array.isArray(ur.data)) window.DB.users = ur.data;
        if (ar && ar.ok && Array.isArray(ar.data)) window.DB.auditoria = ar.data;
        if (mr && mr.ok && mr.data) {
          var m = mr.data;
          if (m.kpis) window.DB.kpis = m.kpis;
          if (m.statusDist) window.DB.statusDist = m.statusDist;
          if (Array.isArray(m.topClientes)) window.DB.topClientes = m.topClientes;
          if (Array.isArray(m.confSeries)) window.DB.confSeries = m.confSeries;
        }
        setDataVersion(function (v) { return v + 1; });
      });
    }

    window.__refreshData = hydrate;
    hydrate();
    return function () { alive = false; window.__refreshData = null; };
  }, [logged]);

  useEffect(() => { document.documentElement.setAttribute('data-theme', theme); }, [theme]);
  useEffect(() => { document.documentElement.setAttribute('data-density', density); }, [density]);

  const showToast = useCallback((msg, type) => {
    setToastState({ msg, type });
    setTimeout(() => setToastState(null), 2800);
  }, []);

  const nav = useCallback((r, arg) => { setRoute(r); setRouteArg(arg || null); }, []);

  const openDoc = useCallback((d) => { setDoc(d); }, []);
  useEffect(() => { window.__openDoc = openDoc; }, [openDoc]);

  const navDoc = useCallback((dir) => {
    const list = window.DB.documents;
    const i = list.findIndex(x => x.id === doc.id);
    const next = list[(i + dir + list.length) % list.length];
    setDoc(next);
  }, [doc]);

  // Cmd+K global
  useEffect(() => {
    const onKey = e => {
      if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) { e.preventDefault(); setCmdOpen(o => !o); }
    };
    window.addEventListener('keydown', onKey); return () => window.removeEventListener('keydown', onKey);
  }, []);

  const openCopilot = (prefill) => { setCopilotOpen(true); setCopilotFocus(true); };

  if (!logged) return <LoginScreen onLogin={() => setLogged(true)} />;

  const screens = {
    painel: <Dashboard nav={nav} openDoc={openDoc} />,
    documentos: <DocumentsScreen openDoc={openDoc} density={density} setDensity={setDensity} initialFilter={routeArg && routeArg.filtro} toast={showToast} />,
    clientes: <ClientsScreen toast={showToast} nav={nav} />,
    schemas: <SchemasScreen toast={showToast} />,
    exportacoes: <ExportScreen toast={showToast} />,
    usuarios: <UsersScreen toast={showToast} />,
    config: <SettingsScreen theme={theme} setTheme={setTheme} density={density} setDensity={setDensity} toast={showToast} />,
  };
  // documentos ocupa altura total; demais rolam
  const fullHeight = route === 'documentos';

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <Sidebar route={route} nav={nav} collapsed={collapsed} setCollapsed={setCollapsed} />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Topbar route={route} onCommand={() => setCmdOpen(true)} onCopilot={() => setCopilotOpen(true)} theme={theme} setTheme={setTheme} density={density} setDensity={setDensity} onLogout={() => setLogged(false)} />
        <main style={{ flex: 1, overflow: fullHeight ? 'hidden' : 'auto', padding: fullHeight ? '22px 26px 0' : '22px 26px 32px', display: 'flex', flexDirection: 'column' }}>
          <div key={route + '_' + dataVersion} style={{ maxWidth: route === 'config' ? 820 : 1320, width: '100%', margin: '0 auto', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            {screens[route]}
          </div>
        </main>
      </div>

      {doc && <ReviewDialog doc={doc} onClose={() => setDoc(null)} onNav={navDoc} onAction={() => {}} toast={showToast} />}
      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} nav={nav} openCopilot={openCopilot} setTheme={setTheme} theme={theme} />
      <CopilotPanel open={copilotOpen} onClose={() => setCopilotOpen(false)} nav={nav} openDoc={openDoc} autoFocus={copilotFocus} />
      <Toast toast={toast} />

      {/* botão flutuante do copiloto */}
      {!copilotOpen && (
        <button onClick={() => setCopilotOpen(true)} title="Copiloto (⌘K)" style={{
          position: 'fixed', bottom: 22, right: 22, width: 48, height: 48, borderRadius: 999, zIndex: 120,
          background: 'var(--accent)', color: '#fff', border: 'none', boxShadow: 'var(--shadow-3)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--accent-hover)'} onMouseLeave={e => e.currentTarget.style.background = 'var(--accent)'}>
          <Icon name="copiloto" size={23} />
        </button>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
