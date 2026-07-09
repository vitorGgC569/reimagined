/* ============================================================
   OContabil — Tela de Login
   ============================================================ */

function Logo({ size = 30, showText = true, color = 'var(--accent)', textColor = 'var(--text)', accentColor = 'var(--accent)' }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
      <span style={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
        <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <rect x="1" y="1" width="30" height="30" rx="7" fill={color} />
          <path d="M9 22V10h2.2v9.9H17V22H9Z" fill="#fff" />
          <circle cx="20.5" cy="12.2" r="2.0" fill="#fff" />
        </svg>
      </span>
      {showText && (
        <span style={{ fontSize: size * 0.6, fontWeight: 600, letterSpacing: '-0.02em', color: textColor }}>
          O<span style={{ color: accentColor }}>Contábil</span>
        </span>
      )}
    </span>
  );
}

function LoginScreen({ onLogin }) {
  const [user, setUser] = useState('');
  const [pass, setPass] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPass, setShowPass] = useState(false);
  const [err, setErr] = useState('');
  // Troca de senha obrigatória (primeiro acesso / MustChangePassword).
  const [pending, setPending] = useState(null);
  const [nova, setNova] = useState('');
  const [conf, setConf] = useState('');
  const [chgErr, setChgErr] = useState('');
  const [chgBusy, setChgBusy] = useState(false);

  const submit = (e) => {
    e.preventDefault();
    setLoading(true); setErr('');
    window.OContabilBridge.call('login', { user: user, pass: pass }).then(function (r) {
      setLoading(false);
      if (r && r.ok) {
        if (r.data && r.data.mustChangePassword) { setPending(r.data); }
        else { onLogin(r.data); }
      }
      else { setErr((r && r.error) || 'Falha no login'); }
    });
  };

  const submitChange = (e) => {
    e.preventDefault();
    setChgErr('');
    if (nova.length < 8) { setChgErr('A nova senha deve ter ao menos 8 caracteres.'); return; }
    if (nova !== conf) { setChgErr('A confirmação não confere.'); return; }
    if (nova === pass) { setChgErr('A nova senha deve ser diferente da temporária.'); return; }
    setChgBusy(true);
    window.OContabilBridge.call('password.change', { atual: pass, nova: nova }).then(function (r) {
      setChgBusy(false);
      if (r && r.ok) { const u = pending; setPending(null); onLogin(u); }
      else { setChgErr((r && r.error) || 'Não foi possível alterar a senha.'); }
    });
  };

  const inputStyle = {
    width: '100%', height: 44, padding: '0 14px', fontSize: 14,
    background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)',
    color: 'var(--text)', outline: 'none', fontFamily: 'var(--font-sans)', transition: 'border-color .14s',
  };

  // Gate obrigatório: primeiro acesso / senha temporária → troca antes de entrar.
  if (pending) {
    return (
      <div style={{ minHeight: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--paper)', padding: 32 }}>
        <form onSubmit={submitChange} style={{ width: '100%', maxWidth: 400 }}>
          <Logo size={30} />
          <h1 style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.02em', margin: '22px 0 6px', color: 'var(--text)' }}>Defina uma nova senha</h1>
          <p style={{ fontSize: 13.5, color: 'var(--text-2)', margin: '0 0 26px' }}>
            Primeiro acesso: por segurança, troque a senha temporária antes de continuar.
          </p>

          <label style={{ display: 'block', fontSize: 12.5, fontWeight: 500, color: 'var(--text-2)', marginBottom: 7 }}>Nova senha (mín. 8)</label>
          <input type="password" value={nova} onChange={e => setNova(e.target.value)} style={inputStyle}
            onFocus={e => e.target.style.borderColor = 'var(--accent)'} onBlur={e => e.target.style.borderColor = 'var(--hairline)'} />

          <label style={{ display: 'block', fontSize: 12.5, fontWeight: 500, color: 'var(--text-2)', margin: '18px 0 7px' }}>Confirmar nova senha</label>
          <input type="password" value={conf} onChange={e => setConf(e.target.value)} style={inputStyle}
            onFocus={e => e.target.style.borderColor = 'var(--accent)'} onBlur={e => e.target.style.borderColor = 'var(--hairline)'} />

          {chgErr && <div style={{ margin: '16px 0 0', fontSize: 12.5, color: 'var(--st-rejected)', background: 'var(--st-rejected-bg)', padding: '9px 12px', borderRadius: 'var(--r-md)', border: '1px solid var(--hairline)' }}>{chgErr}</div>}
          <div style={{ marginTop: 22 }}>
            <Button type="submit" variant="primary" size="lg" full disabled={chgBusy}>
              {chgBusy ? 'Salvando…' : 'Salvar e entrar'}
            </Button>
          </div>
        </form>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100%', display: 'flex', background: 'var(--paper)' }}>
      {/* Painel esquerdo — marca / soberania */}
      <div style={{
        flex: '1 1 46%', background: 'var(--ink)', color: 'var(--paper)', position: 'relative',
        display: 'flex', flexDirection: 'column', justifyContent: 'space-between', padding: '48px 52px',
        overflow: 'hidden',
      }}>
        {/* textura sutil de balancete */}
        <div style={{ position: 'absolute', inset: 0, opacity: 0.05, backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 31px, #FBFAF7 31px, #FBFAF7 32px)' }} />
        <div style={{ position: 'relative', zIndex: 1 }}>
          <Logo size={34} color="var(--accent)" textColor="#FBFAF7" accentColor="#7FCFBF" />
        </div>

        <div style={{ position: 'relative', zIndex: 1, maxWidth: 420 }}>
          <p className="mono" style={{ fontSize: 12, letterSpacing: '0.14em', color: '#7FCFBF', textTransform: 'uppercase', margin: '0 0 18px' }}>Soberania de dados fiscais</p>
          <h2 style={{ fontSize: 30, fontWeight: 600, lineHeight: 1.18, letterSpacing: '-0.02em', margin: 0, color: '#FBFAF7' }}>
            Roda na sua máquina.<br />Nada sai daqui.
          </h2>
          <p style={{ fontSize: 14.5, lineHeight: 1.6, color: '#9DB0A8', margin: '18px 0 0' }}>
            A extração com IA acontece 100% local. Sigilo fiscal e LGPD garantidos por arquitetura — nenhum documento trafega para a nuvem.
          </p>
        </div>

        <div style={{ position: 'relative', zIndex: 1, display: 'flex', gap: 22, flexWrap: 'wrap' }}>
          {[['escudo', '100% local'], ['cadeado', 'LGPD'], ['relogio', 'Trilha de auditoria']].map(([ic, t]) => (
            <span key={t} style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: '#9DB0A8' }}>
              <span style={{ color: '#7FCFBF' }}><Icon name={ic} size={15} /></span>{t}
            </span>
          ))}
        </div>
      </div>

      {/* Painel direito — formulário */}
      <div style={{ flex: '1 1 54%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32 }}>
        <form onSubmit={submit} style={{ width: '100%', maxWidth: 384 }}>
          <h1 style={{ fontSize: 23, fontWeight: 600, letterSpacing: '-0.02em', margin: 0, color: 'var(--text)' }}>Entrar no OContabil</h1>
          <p style={{ fontSize: 13.5, color: 'var(--text-2)', margin: '7px 0 30px' }}>Escritório Razão Contabilidade · estação CAMP-04</p>

          <label style={{ display: 'block', fontSize: 12.5, fontWeight: 500, color: 'var(--text-2)', marginBottom: 7 }}>Usuário</label>
          <input value={user} onChange={e => setUser(e.target.value)} style={inputStyle}
            onFocus={e => e.target.style.borderColor = 'var(--accent)'} onBlur={e => e.target.style.borderColor = 'var(--hairline)'} />

          <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, fontWeight: 500, color: 'var(--text-2)', margin: '18px 0 7px' }}>
            <span>Senha</span>
            <a href="#" onClick={e => e.preventDefault()} style={{ color: 'var(--accent)', textDecoration: 'none', fontWeight: 500 }}>Esqueci a senha</a>
          </label>
          <div style={{ position: 'relative' }}>
            <input type={showPass ? 'text' : 'password'} value={pass} onChange={e => setPass(e.target.value)} style={{ ...inputStyle, paddingRight: 44 }}
              onFocus={e => e.target.style.borderColor = 'var(--accent)'} onBlur={e => e.target.style.borderColor = 'var(--hairline)'} />
            <button type="button" onClick={() => setShowPass(s => !s)} aria-label="Mostrar senha"
              style={{ position: 'absolute', right: 6, top: 6, height: 32, width: 32, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', border: 'none', color: 'var(--text-3)', borderRadius: 5 }}>
              <Icon name="olho" size={17} />
            </button>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 9, margin: '20px 0 26px', fontSize: 13, color: 'var(--text-2)', cursor: 'pointer' }}>
            <input type="checkbox" defaultChecked style={{ width: 16, height: 16, accentColor: 'var(--accent)' }} />
            Manter sessão nesta estação por 8 horas
          </label>

          {err && <div style={{ marginBottom: 14, fontSize: 12.5, color: 'var(--st-rejected)', background: 'var(--st-rejected-bg)', padding: '9px 12px', borderRadius: 'var(--r-md)', border: '1px solid var(--hairline)' }}>{err}</div>}
          <Button type="submit" variant="primary" size="lg" full disabled={loading}>
            {loading ? <span style={{ display: 'inline-flex', alignItems: 'center', gap: 9 }}><Icon name="spinner" size={17} style={{ animation: 'om-spin .8s linear infinite' }} /> Verificando…</span> : 'Entrar'}
          </Button>

          <div style={{ marginTop: 28, paddingTop: 22, borderTop: '1px solid var(--hairline)', display: 'flex', justifyContent: 'center' }}>
            <LocalSeal />
          </div>
          <p style={{ textAlign: 'center', fontSize: 11.5, color: 'var(--text-3)', margin: '20px 0 0' }} className="mono">OContabil v3.2.1 · build local · ONNX Runtime</p>
        </form>
      </div>
    </div>
  );
}

window.Logo = Logo;
window.LoginScreen = LoginScreen;
