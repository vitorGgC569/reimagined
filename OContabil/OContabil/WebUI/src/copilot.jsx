/* ============================================================
   OContabil — Copiloto local (Cmd+K + painel lateral)
   Assistente que AGE sobre os dados, mostrando as ferramentas
   executadas e resultados ancorados (rastro). Tom sóbrio.
   ============================================================ */

/* ---- Motor de respostas (ancorado nos dados reais) ---- */
function responder(q, ctx) {
  const DB = window.DB;
  const ql = q.toLowerCase();

  // parados há mais de N dias
  if (/parado|paradas|travad|atrasad/.test(ql)) {
    const parados = DB.documents.filter(d => d.diasParado != null).sort((a, b) => b.diasParado - a.diasParado);
    return {
      tools: [{ nome: 'consultar_documentos', args: 'status=pendente,revisar · dias_parado>3', n: parados.length }],
      texto: `Encontrei **${parados.length} documentos parados há mais de 3 dias**. O mais antigo está há ${parados[0]?.diasParado || 0} dias. Eles somam **R$ ${DB.brl(parados.reduce((s, d) => s + d.valor, 0))}** em valor.`,
      tabela: { cols: ['Tipo / Nº', 'Cliente', 'Valor', 'Parado', 'Status'], rows: parados.slice(0, 6).map(d => ({ doc: d, cells: [d.tipo + ' ' + d.numero, d.cliente, 'R$ ' + DB.brl(d.valor), d.diasParado + 'd', d.status] })) },
      acoes: [{ label: 'Abrir lista filtrada', icon: 'filtro', go: 'documentos', arg: 'parados' }],
    };
  }
  // notas de cliente acima de valor
  if (/(nota|notas|nf|documento).*(acima|maior|>|mais de)|empresa|cliente.*r\$|marília|marilia|horizonte/.test(ql)) {
    const cli = DB.clients.find(c => ql.includes(c.fantasia.toLowerCase().split(' ')[0]) || ql.includes(c.nome.toLowerCase().split(' ')[0])) || DB.clients[0];
    const limite = (ql.match(/r?\$?\s*([\d.]+)\s*mil/) ? parseFloat(RegExp.$1) * 1000 : (ql.match(/([\d.]+)\.?000/) ? 5000 : 5000));
    const res = DB.documents.filter(d => d.clienteId === cli.id && d.valor >= limite).sort((a, b) => b.valor - a.valor);
    return {
      tools: [
        { nome: 'buscar_cliente', args: 'nome~"' + cli.fantasia + '"', n: 1 },
        { nome: 'consultar_documentos', args: 'cliente=' + cli.id + ' · valor>=' + DB.brl(limite), n: res.length },
      ],
      texto: `**${res.length} documentos** de **${cli.fantasia}** acima de R$ ${DB.brl(limite)}, totalizando **R$ ${DB.brl(res.reduce((s, d) => s + d.valor, 0))}**.`,
      tabela: { cols: ['Tipo / Nº', 'Valor', 'Data', 'Status'], rows: res.slice(0, 6).map(d => ({ doc: d, cells: [d.tipo + ' ' + d.numero, 'R$ ' + DB.brl(d.valor), d.dataStr, d.status] })) },
      acoes: [{ label: 'Ver no painel de documentos', icon: 'documentos', go: 'documentos' }, { label: 'Exportar seleção', icon: 'download', go: 'exportacoes' }],
    };
  }
  // exportar DARFs / SPED
  if (/export|sped|domínio|dominio|darf|trimestre/.test(ql)) {
    const darfs = DB.documents.filter(d => d.tipo === 'DARF');
    const fmt = /sped/.test(ql) ? 'SPED Fiscal' : /domínio|dominio/.test(ql) ? 'Domínio Sistemas' : 'SPED Fiscal';
    return {
      tools: [
        { nome: 'consultar_documentos', args: 'tipo=DARF · período=2º trimestre', n: darfs.length },
        { nome: 'preparar_exportacao', args: 'formato=' + fmt, n: darfs.length },
      ],
      texto: `Preparei a exportação de **${darfs.length} DARFs** do trimestre para **${fmt}**. Valor total **R$ ${DB.brl(darfs.reduce((s, d) => s + d.valor, 0))}**. Revise antes de confirmar — nada é enviado para fora desta máquina.`,
      tabela: { cols: ['Nº', 'Cliente', 'Valor', 'Status'], rows: darfs.slice(0, 5).map(d => ({ doc: d, cells: [d.numero, d.cliente, 'R$ ' + DB.brl(d.valor), d.status] })) },
      acoes: [{ label: 'Abrir exportação ' + fmt, icon: 'download', go: 'exportacoes' }],
    };
  }
  // confiança / qualidade
  if (/confian|qualidade|baixa|revisar/.test(ql)) {
    const baixa = DB.documents.filter(d => d.confianca != null && d.confianca < 65);
    return {
      tools: [{ nome: 'consultar_documentos', args: 'confianca<65%', n: baixa.length }],
      texto: `Há **${baixa.length} documentos com confiança abaixo de 65%** que merecem revisão prioritária. A confiança média da base hoje é **${DB.kpis.confiancaMedia}%**.`,
      tabela: { cols: ['Tipo / Nº', 'Cliente', 'Confiança', 'Status'], rows: baixa.slice(0, 6).map(d => ({ doc: d, cells: [d.tipo + ' ' + d.numero, d.cliente, d.confianca + '%', d.status] })) },
      acoes: [{ label: 'Revisar de baixa confiança', icon: 'olho', go: 'documentos' }],
    };
  }
  // resumo / total do dia
  if (/quantos|total|resumo|hoje|processad/.test(ql)) {
    return {
      tools: [{ nome: 'agregar_kpis', args: 'data=hoje', n: 1 }],
      texto: `Hoje foram **${DB.kpis.processadosHoje} documentos processados**, com **${DB.kpis.naFila} na fila** e **${DB.kpis.aguardandoRevisao} aguardando revisão**. Confiança média de **${DB.kpis.confiancaMedia}%**.`,
      acoes: [{ label: 'Abrir painel', icon: 'painel', go: 'painel' }],
    };
  }
  // fallback
  return {
    tools: [{ nome: 'interpretar_pergunta', args: '"' + q + '"', n: 0 }],
    texto: 'Posso consultar, filtrar e exportar documentos, calcular totais e localizar pendências — tudo localmente. Tente, por exemplo, *"quantos documentos estão parados há mais de 3 dias?"* ou *"exporta os DARFs do trimestre pro SPED"*.',
    acoes: [],
  };
}

const SUGESTOES = [
  'Quantos documentos estão parados há mais de 3 dias?',
  'Me traz as notas do Mercado Marília acima de R$ 5.000',
  'Exporta os DARFs do trimestre pro SPED',
  'Quais documentos estão com baixa confiança?',
];

/* ---- Chip de ferramenta executada ---- */
function ToolChip({ tool }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '7px 11px', background: 'var(--surface-2)', border: '1px solid var(--hairline-2)', borderRadius: 'var(--r-md)', fontSize: 12 }}>
      <span style={{ color: 'var(--accent)', display: 'inline-flex' }}><Icon name="ferramenta" size={14} /></span>
      <span className="mono" style={{ color: 'var(--text)', fontWeight: 500 }}>{tool.nome}</span>
      <span className="mono" style={{ color: 'var(--text-3)', fontSize: 11, flex: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{tool.args}</span>
      <span style={{ color: 'var(--st-approved)', display: 'inline-flex' }}><Icon name="check" size={13} /></span>
    </div>
  );
}

/* ---- Renderiza markdown leve (negrito/itálico) ---- */
function MdText({ text }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return <span>{parts.map((p, i) => {
    if (p.startsWith('**')) return <b key={i} style={{ color: 'var(--text)', fontWeight: 600 }}>{p.slice(2, -2)}</b>;
    if (p.startsWith('*')) return <i key={i} style={{ color: 'var(--text-2)' }}>{p.slice(1, -1)}</i>;
    return p;
  })}</span>;
}

/* ---- Painel do Copiloto ---- */
function CopilotPanel({ open, onClose, nav, openDoc, autoFocus }) {
  const DB = window.DB;
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState('');
  const [thinking, setThinking] = useState(false);
  const inputRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => { if (open && autoFocus && inputRef.current) setTimeout(() => inputRef.current.focus(), 60); }, [open, autoFocus]);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [msgs, thinking]);

  const send = (q) => {
    const pergunta = (q != null ? q : input).trim();
    if (!pergunta) return;
    setMsgs(m => [...m, { role: 'user', text: pergunta }]);
    setInput(''); setThinking(true);
    setTimeout(() => {
      const r = responder(pergunta, {});
      setThinking(false);
      setMsgs(m => [...m, { role: 'assistant', ...r }]);
    }, 720);
  };

  return (
    <div style={{
      position: 'fixed', top: 0, right: 0, bottom: 0, width: 420, maxWidth: '100vw', zIndex: 180,
      background: 'var(--surface)', borderLeft: '1px solid var(--hairline)', boxShadow: 'var(--shadow-3)',
      display: 'flex', flexDirection: 'column', transform: open ? 'translateX(0)' : 'translateX(100%)',
      transition: 'transform .26s cubic-bezier(.2,.8,.2,1)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 11, padding: '14px 16px', borderBottom: '1px solid var(--hairline)', flexShrink: 0 }}>
        <span style={{ color: 'var(--accent)', display: 'inline-flex', padding: 7, background: 'var(--accent-weak)', borderRadius: 'var(--r-md)' }}><Icon name="copiloto" size={19} /></span>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <h3 style={{ margin: 0, fontSize: 14.5, fontWeight: 600 }}>Copiloto</h3>
            <LocalSeal variant="compact" />
          </div>
          <p style={{ margin: '2px 0 0', fontSize: 11.5, color: 'var(--text-2)' }}>Assistente fiscal · roda nesta máquina</p>
        </div>
        <button onClick={onClose} style={miniBtn}><Icon name="x" size={16} /></button>
      </div>

      {/* Thread */}
      <div ref={scrollRef} style={{ flex: 1, overflow: 'auto', padding: 16 }}>
        {msgs.length === 0 && (
          <div style={{ animation: 'om-fade-in .3s ease' }}>
            <div style={{ textAlign: 'center', padding: '24px 8px 18px' }}>
              <span style={{ color: 'var(--accent)', display: 'inline-flex', padding: 13, background: 'var(--accent-weak)', borderRadius: 'var(--r-lg)', marginBottom: 14 }}><Icon name="copiloto" size={26} /></span>
              <h4 style={{ margin: 0, fontSize: 15, fontWeight: 600 }}>Pergunte sobre seus documentos</h4>
              <p style={{ margin: '6px auto 0', fontSize: 12.5, color: 'var(--text-2)', maxWidth: 280 }}>O copiloto consulta, filtra e exporta agindo sobre os dados reais — e mostra cada ferramenta que executou.</p>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8 }}>
              {SUGESTOES.map(s => (
                <button key={s} onClick={() => send(s)} style={{
                  display: 'flex', alignItems: 'center', gap: 10, textAlign: 'left', padding: '11px 13px', fontSize: 13,
                  background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)',
                }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--zebra)'; e.currentTarget.style.borderColor = 'var(--accent-weak-2)'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'var(--surface)'; e.currentTarget.style.borderColor = 'var(--hairline)'; }}>
                  <span style={{ color: 'var(--accent)', display: 'inline-flex' }}><Icon name="raio" size={15} /></span>
                  <span style={{ flex: 1 }}>{s}</span>
                  <span style={{ color: 'var(--text-3)' }}><Icon name="setaDir" size={14} /></span>
                </button>
              ))}
            </div>
          </div>
        )}

        {msgs.map((m, i) => (
          <div key={i} style={{ marginBottom: 18, animation: 'om-pop-in .2s ease' }}>
            {m.role === 'user' ? (
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{ maxWidth: '85%', padding: '9px 13px', background: 'var(--accent-weak)', border: '1px solid var(--accent-weak-2)', borderRadius: '12px 12px 4px 12px', fontSize: 13, color: 'var(--accent-ink)' }}>{m.text}</div>
              </div>
            ) : (
              <div>
                {m.tools && m.tools.length > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 11 }}>
                    <span style={{ fontSize: 10.5, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>Ferramentas executadas</span>
                    {m.tools.map((t, j) => <ToolChip key={j} tool={t} />)}
                  </div>
                )}
                <div style={{ fontSize: 13.5, lineHeight: 1.55, color: 'var(--text-2)' }}><MdText text={m.texto} /></div>
                {m.tabela && (
                  <div style={{ marginTop: 12, border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', overflow: 'hidden' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5 }}>
                      <thead><tr style={{ background: 'var(--surface-2)' }}>
                        {m.tabela.cols.map((c, j) => <th key={j} style={{ padding: '7px 10px', textAlign: j === 0 ? 'left' : (j === m.tabela.cols.length - 1 ? 'left' : 'right'), fontSize: 10, fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: '1px solid var(--hairline)' }}>{c}</th>)}
                      </tr></thead>
                      <tbody>
                        {m.tabela.rows.map((r, j) => (
                          <tr key={j} onClick={() => r.doc && openDoc(r.doc)} style={{ cursor: r.doc ? 'pointer' : 'default', borderBottom: j < m.tabela.rows.length - 1 ? '1px solid var(--hairline-2)' : 'none' }}
                            onMouseEnter={e => e.currentTarget.style.background = 'var(--zebra)'} onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                            {r.cells.map((cell, k) => {
                              const isStatus = k === r.cells.length - 1 && window.DB.STATUS[cell];
                              return (
                                <td key={k} className={k > 0 && k < r.cells.length - 1 ? 'mono' : ''} style={{ padding: '7px 10px', textAlign: k === 0 || isStatus ? 'left' : 'right', color: k === 0 ? 'var(--text)' : 'var(--text-2)', fontWeight: k === 0 ? 500 : 400, whiteSpace: 'nowrap' }}>
                                  {isStatus ? <StatusBadge status={cell} size="sm" /> : cell}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {m.acoes && m.acoes.length > 0 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
                    {m.acoes.map((a, j) => (
                      <Button key={j} size="sm" variant={j === 0 ? 'primary' : 'default'} icon={a.icon} onClick={() => { nav(a.go, a.arg ? { filtro: a.arg } : undefined); onClose(); }}>{a.label}</Button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {thinking && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 12.5, color: 'var(--text-2)', padding: '4px 0' }}>
            <span style={{ display: 'inline-flex', gap: 3 }}>
              {[0, 1, 2].map(i => <span key={i} style={{ width: 6, height: 6, borderRadius: 999, background: 'var(--accent)', animation: 'om-pulse 1s ease-in-out infinite', animationDelay: i * 0.15 + 's' }} />)}
            </span>
            Consultando dados locais…
          </div>
        )}
      </div>

      {/* Input */}
      <div style={{ padding: 14, borderTop: '1px solid var(--hairline)', flexShrink: 0 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)} rows={1}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Pergunte ao copiloto…"
            style={{ flex: 1, minHeight: 40, maxHeight: 120, padding: '10px 12px', fontSize: 13.5, fontFamily: 'var(--font-sans)', background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)', outline: 'none', resize: 'none', lineHeight: 1.4 }}
            onFocus={e => e.target.style.borderColor = 'var(--accent)'} onBlur={e => e.target.style.borderColor = 'var(--hairline)'} />
          <Button variant="primary" icon="enviar" onClick={() => send()} disabled={!input.trim()} style={{ height: 40, width: 40, padding: 0 }} />
        </div>
        <p style={{ margin: '9px 2px 0', fontSize: 10.5, color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <Icon name="escudo" size={11} /> Respostas ancoradas nos seus dados · processado 100% local
        </p>
      </div>
    </div>
  );
}

window.CopilotPanel = CopilotPanel;
window.copilotResponder = responder;
