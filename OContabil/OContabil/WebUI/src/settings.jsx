/* ============================================================
   OContabil — Configurações
   ============================================================ */

function Toggle({ checked, onChange }) {
  return (
    <button onClick={() => onChange(!checked)} role="switch" aria-checked={checked} style={{
      width: 42, height: 24, borderRadius: 999, border: 'none', padding: 2, flexShrink: 0,
      background: checked ? 'var(--accent)' : 'var(--hairline)', transition: 'background .18s', position: 'relative',
    }}>
      <span style={{ display: 'block', width: 20, height: 20, borderRadius: 999, background: '#fff', boxShadow: 'var(--shadow-1)', transform: checked ? 'translateX(18px)' : 'translateX(0)', transition: 'transform .18s cubic-bezier(.2,.8,.2,1)' }} />
    </button>
  );
}

function SettingRow({ title, desc, children, last }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 24, padding: '16px 0', borderBottom: last ? 'none' : '1px solid var(--hairline-2)' }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13.5, fontWeight: 500, color: 'var(--text)' }}>{title}</div>
        {desc && <div style={{ fontSize: 12.5, color: 'var(--text-2)', marginTop: 3, maxWidth: 460 }}>{desc}</div>}
      </div>
      <div style={{ flexShrink: 0 }}>{children}</div>
    </div>
  );
}

function SettingsScreen({ theme, setTheme, density, setDensity, toast }) {
  const [limite, setLimite] = useState(3);
  const [backup, setBackup] = useState(true);
  const [motor, setMotor] = useState('onnx');
  const [timeout, setTimeoutV] = useState('8h');
  const [autoAprovar, setAutoAprovar] = useState(true);

  const cardTitle = { margin: '0 0 2px', fontSize: 14.5, fontWeight: 600 };
  const cardSub = { margin: '0 0 6px', fontSize: 12.5, color: 'var(--text-2)' };

  return (
    <div style={{ animation: 'om-fade-in .25s ease', maxWidth: 780 }}>
      <SectionHeader title="Configurações" subtitle="Preferências da estação e do motor de extração" icon="config" />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Alertas */}
        <Panel style={{ padding: '18px 22px' }}>
          <h3 style={cardTitle}>Alertas e limites</h3>
          <p style={cardSub}>Quando documentos ficam parados além do limite, o painel exibe o banner âmbar.</p>
          <SettingRow title="Limite de documentos parados" desc="Dias até um documento pendente/em revisão ser marcado como parado.">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button onClick={() => setLimite(l => Math.max(1, l - 1))} style={{ ...miniBtn, width: 30, height: 30 }}><Icon name="chevDown" size={15} /></button>
              <span className="mono" style={{ fontSize: 16, fontWeight: 600, minWidth: 56, textAlign: 'center' }}>{limite} dias</span>
              <button onClick={() => setLimite(l => l + 1)} style={{ ...miniBtn, width: 30, height: 30 }}><Icon name="chevDown" size={15} style={{ transform: 'rotate(180deg)' }} /></button>
            </div>
          </SettingRow>
          <SettingRow title="Aprovação automática de alta confiança" desc="Documentos com confiança ≥ 98% e sem campos críticos são aprovados automaticamente." last>
            <Toggle checked={autoAprovar} onChange={setAutoAprovar} />
          </SettingRow>
        </Panel>

        {/* Motor de IA */}
        <Panel style={{ padding: '18px 22px' }}>
          <h3 style={cardTitle}>Motor de extração</h3>
          <p style={cardSub}>Todo o processamento ocorre nesta máquina — nenhum dado trafega para a nuvem.</p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, margin: '8px 0 6px' }}>
            {[
              { id: 'onnx', nome: 'ONNX Runtime', desc: 'Rede neural local · recomendado', icon: 'raio' },
              { id: 'python', nome: 'Python / OCR', desc: 'Tesseract + heurística', icon: 'config' },
              { id: 'regex', nome: 'Regras (regex)', desc: 'Determinístico, sem IA', icon: 'schemas' },
            ].map(m => (
              <button key={m.id} onClick={() => setMotor(m.id)} style={{
                textAlign: 'left', padding: 13, borderRadius: 'var(--r-md)', cursor: 'pointer',
                background: motor === m.id ? 'var(--accent-weak)' : 'var(--surface)',
                border: '1px solid ' + (motor === m.id ? 'var(--accent)' : 'var(--hairline)'),
              }}>
                <span style={{ color: motor === m.id ? 'var(--accent)' : 'var(--text-3)', display: 'inline-flex', marginBottom: 9 }}><Icon name={m.icon} size={18} /></span>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{m.nome}</div>
                <div style={{ fontSize: 11, color: 'var(--text-2)', marginTop: 2 }}>{m.desc}</div>
              </button>
            ))}
          </div>
          <div style={{ marginTop: 10 }}><LocalSeal /></div>
        </Panel>

        {/* Backup */}
        <Panel style={{ padding: '18px 22px' }}>
          <h3 style={cardTitle}>Backup e dados</h3>
          <SettingRow title="Backup automático diário" desc="Cópia local criptografada às 03:00 · último: 01/06/2026 03:00 (2,4 GB).">
            <Toggle checked={backup} onChange={setBackup} />
          </SettingRow>
          <SettingRow title="Local do backup" desc="Pasta nesta estação onde os backups são gravados.">
            <span className="mono" style={{ fontSize: 12, color: 'var(--text-2)', background: 'var(--subtle)', padding: '6px 10px', borderRadius: 5 }}>D:\OContabil\backups</span>
          </SettingRow>
          <SettingRow title="Tempo de sessão" desc="Encerra a sessão automaticamente após inatividade." last>
            <Select value={timeout} onChange={setTimeoutV} width={140} options={[{ value: '1h', label: '1 hora' }, { value: '4h', label: '4 horas' }, { value: '8h', label: '8 horas' }, { value: 'never', label: 'Sem limite' }]} />
          </SettingRow>
        </Panel>

        {/* Aparência */}
        <Panel style={{ padding: '18px 22px' }}>
          <h3 style={cardTitle}>Aparência</h3>
          <SettingRow title="Tema" desc="Modo claro recomendado para ambiente de trabalho diurno.">
            <Segmented value={theme} onChange={setTheme} options={[{ value: 'light', icon: 'sol', label: 'Claro' }, { value: 'dark', icon: 'lua', label: 'Escuro' }]} />
          </SettingRow>
          <SettingRow title="Densidade das tabelas" desc="Compacta exibe mais linhas por tela, no estilo planilha." last>
            <Segmented value={density} onChange={setDensity} options={[{ value: 'comfortable', label: 'Confortável' }, { value: 'compact', label: 'Compacta' }]} />
          </SettingRow>
        </Panel>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, paddingBottom: 8 }}>
          <Button variant="default">Restaurar padrões</Button>
          <Button variant="primary" icon="check" onClick={() => toast('Configurações salvas')}>Salvar configurações</Button>
        </div>
      </div>
    </div>
  );
}

window.SettingsScreen = SettingsScreen;
window.Toggle = Toggle;
