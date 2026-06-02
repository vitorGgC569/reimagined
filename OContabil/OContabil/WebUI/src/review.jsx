/* ============================================================
   OContabil — Diálogo de Revisão de Documento (o coração)
   Painel Original (persistente) + abas Campos / Histórico
   ============================================================ */

/* ---- Validação de formato pt-BR ---- */
function validar(tipoVal, valor) {
  if (valor == null || valor === '') return { ok: false, msg: 'Campo vazio' };
  if (tipoVal === 'cnpj') return { ok: /^\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}$/.test(valor), msg: 'CNPJ inválido (00.000.000/0000-00)' };
  if (tipoVal === 'data') return { ok: /^\d{2}\/\d{2}\/\d{4}$/.test(valor), msg: 'Data inválida (dd/mm/aaaa)' };
  if (tipoVal === 'moeda') return { ok: /^\d{1,3}(\.\d{3})*,\d{2}$/.test(valor) || /^\d+,\d{2}$/.test(valor), msg: 'Valor inválido (0,00)' };
  if (tipoVal === 'chave') return { ok: valor.replace(/\s/g, '').length === 44, msg: 'Chave deve ter 44 dígitos' };
  return { ok: true };
}

/* ---- Preview do documento original (placeholder fiscal) ---- */
function DocPreview({ doc }) {
  const DB = window.DB;
  const [zoom, setZoom] = useState(1);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--subtle)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '9px 14px', borderBottom: '1px solid var(--hairline)', background: 'var(--surface-2)' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, fontSize: 12, color: 'var(--text-2)', fontWeight: 500 }}>
          <Icon name="arquivo" size={14} /> Original · {doc.tipo}_{doc.numero}.pdf
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button onClick={() => setZoom(z => Math.max(0.6, z - 0.1))} style={miniBtn}><Icon name="chevDown" size={14} /></button>
          <span className="mono" style={{ fontSize: 11, color: 'var(--text-2)', minWidth: 38, textAlign: 'center' }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(1.6, z + 0.1))} style={miniBtn}><Icon name="chevDown" size={14} style={{ transform: 'rotate(180deg)' }} /></button>
          <div style={{ width: 1, height: 16, background: 'var(--hairline)', margin: '0 4px' }} />
          <button style={miniBtn} title="Baixar original"><Icon name="download" size={14} /></button>
        </div>
      </div>
      <div style={{ flex: 1, overflow: 'auto', padding: 24, display: 'flex', justifyContent: 'center', alignItems: 'flex-start' }}>
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top center', transition: 'transform .15s' }}>
          {/* Folha do documento — representação fiscal estilizada (placeholder) */}
          <div style={{ width: 420, background: '#fff', boxShadow: 'var(--shadow-2)', borderRadius: 2, color: '#16201C', fontFamily: 'var(--font-mono)', fontSize: 9.5, lineHeight: 1.5 }}>
            <div style={{ borderBottom: '2px solid #16201C', padding: '14px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: 11, fontFamily: 'var(--font-sans)' }}>DANFE</div>
                <div style={{ fontSize: 7.5, color: '#555' }}>Documento Auxiliar da Nota Fiscal Eletrônica</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div style={{ fontWeight: 700 }}>Nº {doc.numero}</div>
                <div>SÉRIE {doc.serie}</div>
              </div>
            </div>
            <div style={{ padding: '10px 16px', borderBottom: '1px solid #ccc' }}>
              <div style={{ fontSize: 7, color: '#888', letterSpacing: '0.05em' }}>CHAVE DE ACESSO</div>
              <div style={{ letterSpacing: '0.5px', wordBreak: 'break-all', marginTop: 2 }}>{DB.chaveFmt(doc.chave)}</div>
            </div>
            <div style={{ padding: '10px 16px', borderBottom: '1px solid #ccc' }}>
              <div style={{ fontSize: 7, color: '#888' }}>EMITENTE</div>
              <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: 10, marginTop: 1 }}>{doc.emitente}</div>
              <div style={{ marginTop: 3 }}>CNPJ 34.115.890/0001-72 · Av. Industrial, 1.840 · {doc.cliente.split(' ')[0]}</div>
            </div>
            <div style={{ padding: '10px 16px', borderBottom: '1px solid #ccc' }}>
              <div style={{ fontSize: 7, color: '#888' }}>DESTINATÁRIO</div>
              <div style={{ fontFamily: 'var(--font-sans)', fontWeight: 600, fontSize: 10, marginTop: 1 }}>{doc.cliente}</div>
              <div style={{ marginTop: 3 }}>CNPJ 12.345.678/0001-90</div>
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 8 }}>
              <thead><tr style={{ background: '#f0f0f0' }}>
                <td style={{ padding: '4px 16px', fontWeight: 700 }}>DESCRIÇÃO</td>
                <td style={{ padding: '4px 8px', fontWeight: 700, textAlign: 'right' }}>QTD</td>
                <td style={{ padding: '4px 16px', fontWeight: 700, textAlign: 'right' }}>VALOR</td>
              </tr></thead>
              <tbody>
                {['Mercadoria conforme pedido', 'Frete CIF', 'Serviço de instalação'].map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '4px 16px' }}>{p}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{i + 1}</td>
                    <td style={{ padding: '4px 16px', textAlign: 'right' }}>{DB.brl(doc.valor / (3 - i * 0.4))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ padding: '10px 16px', display: 'flex', justifyContent: 'space-between', borderTop: '2px solid #16201C', fontWeight: 700, fontSize: 11, fontFamily: 'var(--font-sans)' }}>
              <span>VALOR TOTAL</span><span>R$ {DB.brl(doc.valor)}</span>
            </div>
            <div style={{ padding: '8px 16px 16px', fontSize: 7, color: '#999' }}>Emissão {doc.dataStr} · Protocolo de autorização 135{doc.numero}0014820 · Recebemos os produtos constantes da NF-e indicada ao lado.</div>
          </div>
        </div>
      </div>
    </div>
  );
}
const miniBtn = { width: 26, height: 26, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 5, color: 'var(--text-2)' };

/* ---- Campo editável ---- */
function FieldRow({ campo, onChange, onEdit }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(campo.valor);
  const inputRef = useRef(null);
  const low = campo.conf < 75;
  const c = window.confColor(campo.conf);
  const v = validar(campo.tipoVal, val);

  useEffect(() => { if (editing && inputRef.current) inputRef.current.focus(); }, [editing]);

  const commit = () => {
    setEditing(false);
    if (val !== campo.valor) { onChange(val); onEdit && onEdit(campo, campo.valor, val); }
  };

  return (
    <div style={{
      padding: '11px 14px', borderRadius: 'var(--r-md)', position: 'relative',
      gridColumn: campo.span === 2 ? '1 / -1' : 'auto',
      background: low ? 'var(--st-review-bg)' : 'var(--surface)',
      border: '1px solid ' + (low ? 'var(--st-review)' + '40' : 'var(--hairline-2)'),
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontSize: 11.5, color: 'var(--text-2)', fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          {campo.campo}
          {low && <Tip label="Baixa confiança — revise este campo"><span style={{ color: 'var(--st-review)', display: 'inline-flex' }}><Icon name="alerta" size={12} /></span></Tip>}
        </span>
        <ConfidencePill value={campo.conf} />
      </div>
      {editing ? (
        <div>
          <input ref={inputRef} value={val} onChange={e => setVal(e.target.value)}
            onBlur={commit} onKeyDown={e => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') { setVal(campo.valor); setEditing(false); } }}
            className={campo.tipoVal !== 'texto' ? 'mono' : ''}
            style={{ width: '100%', height: 32, padding: '0 10px', fontSize: 13, background: 'var(--surface)', border: '1px solid ' + (v.ok ? 'var(--accent)' : 'var(--st-rejected)'), borderRadius: 5, color: 'var(--text)', outline: 'none' }} />
          {!v.ok && <span style={{ fontSize: 10.5, color: 'var(--st-rejected)', marginTop: 4, display: 'block' }}>{v.msg}</span>}
        </div>
      ) : (
        <button onClick={() => setEditing(true)} style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, width: '100%', textAlign: 'left',
          background: 'transparent', border: 'none', padding: 0, color: 'var(--text)', cursor: 'text', minHeight: 22,
        }}
          onMouseEnter={e => { const ed = e.currentTarget.querySelector('.edit-ic'); if (ed) ed.style.opacity = 1; }}
          onMouseLeave={e => { const ed = e.currentTarget.querySelector('.edit-ic'); if (ed) ed.style.opacity = 0; }}>
          <span className={campo.tipoVal !== 'texto' ? 'mono' : ''} style={{ fontSize: 13.5, fontWeight: 500, wordBreak: campo.tipoVal === 'chave' ? 'break-all' : 'normal', lineHeight: 1.4 }}>
            {campo.tipoVal === 'moeda' ? 'R$ ' + val : campo.tipoVal === 'chave' ? window.DB.chaveFmt(val.replace(/\s/g, '')) : val}
          </span>
          <span className="edit-ic" style={{ opacity: 0, transition: 'opacity .12s', color: 'var(--text-3)', flexShrink: 0 }}><Icon name="lapis" size={13} /></span>
        </button>
      )}
    </div>
  );
}

/* ---- Linha de histórico/auditoria ---- */
function HistoryItem({ item, last }) {
  const map = {
    sistema: { ic: 'raio', c: 'var(--st-info)' }, edicao: { ic: 'lapis', c: 'var(--st-review)' },
    aprovado: { ic: 'check', c: 'var(--st-approved)' }, rejeitado: { ic: 'x', c: 'var(--st-rejected)' },
  };
  const m = map[item.tipo] || map.sistema;
  return (
    <div style={{ display: 'flex', gap: 13, position: 'relative' }}>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
        <span style={{ width: 28, height: 28, borderRadius: 999, background: m.c + '18', color: m.c, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', border: '1px solid ' + m.c + '30' }}><Icon name={m.ic} size={14} /></span>
        {!last && <span style={{ width: 1.5, flex: 1, background: 'var(--hairline)', marginTop: 4, minHeight: 18 }} />}
      </div>
      <div style={{ paddingBottom: last ? 0 : 20, flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{item.acao}</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--text-3)', whiteSpace: 'nowrap' }}>{item.quando}</span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 3 }}>{item.detalhe}</div>
        <div style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 5, display: 'inline-flex', alignItems: 'center', gap: 5 }}><Icon name="usuarios" size={12} /> {item.quem}</div>
      </div>
    </div>
  );
}

/* ---- Diálogo principal ---- */
function ReviewDialog({ doc, onClose, onNav, onAction, toast }) {
  const DB = window.DB;
  const [tab, setTab] = useState('campos');
  const [campos, setCampos] = useState(() => DB.camposDe(doc).map(c => ({ ...c })));
  const [extraHist, setExtraHist] = useState([]);
  const [rejectOpen, setRejectOpen] = useState(false);
  const [rejectReason, setRejectReason] = useState('');
  const [resolved, setResolved] = useState(null); // 'aprovado' | 'rejeitado'

  useEffect(() => {
    setCampos(DB.camposDe(doc).map(c => ({ ...c })));
    setExtraHist([]); setResolved(null); setTab('campos');
  }, [doc.id]);

  const lowCount = campos.filter(c => c.conf < 75).length;

  const editField = (campo, oldV, newV) => {
    setCampos(cs => cs.map(c => c.campo === campo.campo ? { ...c, valor: newV, conf: 100, edited: true } : c));
    setExtraHist(h => [...h, { quem: 'Você (Ana B. Souza)', acao: 'Campo corrigido', detalhe: campo.campo + ': "' + oldV + '" → "' + newV + '"', quando: '01/06/2026 ' + new Date().toTimeString().slice(0, 5), tipo: 'edicao' }]);
  };

  const aprovar = () => { setResolved('aprovado'); onAction && onAction(doc, 'aprovado'); toast('✓ ' + doc.tipo + ' ' + doc.numero + ' aprovado'); setTimeout(onClose, 650); };
  const confirmReject = () => {
    if (!rejectReason.trim()) return;
    setResolved('rejeitado'); setRejectOpen(false); onAction && onAction(doc, 'rejeitado', rejectReason);
    toast('Documento ' + doc.numero + ' rejeitado'); setTimeout(onClose, 650);
  };

  // atalhos de teclado
  useEffect(() => {
    const onKey = e => {
      if (rejectOpen) return;
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'a' || e.key === 'A') aprovar();
      else if (e.key === 'r' || e.key === 'R') setRejectOpen(true);
      else if (e.key === 'ArrowRight') onNav(1);
      else if (e.key === 'ArrowLeft') onNav(-1);
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [doc.id, rejectOpen]);

  const hist = [...DB.historicoDe(doc), ...extraHist];

  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 250, background: 'rgba(22,32,28,0.5)', backdropFilter: 'blur(2px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, animation: 'om-fade-in .15s ease' }}>
      <div onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" style={{
        width: 1180, maxWidth: '100%', height: '90vh', background: 'var(--surface)', borderRadius: 'var(--r-lg)',
        boxShadow: 'var(--shadow-3)', border: '1px solid var(--hairline)', display: 'flex', flexDirection: 'column',
        overflow: 'hidden', animation: 'om-pop-in .2s cubic-bezier(.2,.8,.2,1)',
      }}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '14px 18px', borderBottom: '1px solid var(--hairline)', flexShrink: 0 }}>
          <span style={{ color: 'var(--accent)', display: 'inline-flex', padding: 8, background: 'var(--accent-weak)', borderRadius: 'var(--r-md)' }}><Icon name={window.DOC_ICON[doc.tipo] || 'arquivo'} size={20} /></span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{doc.tipo} <span className="mono">{doc.numero}</span></h2>
              <StatusBadge status={resolved || doc.status} size="sm" />
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 2 }}>{doc.cliente} · {doc.emitente} · <span className="mono">R$ {DB.brl(doc.valor)}</span></div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button onClick={() => onNav(-1)} title="Anterior (←)" style={miniBtn}><Icon name="chevLeft" size={16} /></button>
            <button onClick={() => onNav(1)} title="Próximo (→)" style={miniBtn}><Icon name="chevRight" size={16} /></button>
            <button onClick={onClose} title="Fechar (Esc)" style={{ ...miniBtn, width: 32, height: 32 }}><Icon name="x" size={17} /></button>
          </div>
        </div>

        {/* Corpo: Original | abas */}
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          {/* Original — persistente */}
          <div style={{ width: '44%', borderRight: '1px solid var(--hairline)', minWidth: 0 }}>
            <DocPreview doc={doc} />
          </div>

          {/* Direita */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <div style={{ display: 'flex', gap: 2, padding: '10px 16px 0', borderBottom: '1px solid var(--hairline)', flexShrink: 0 }}>
              {[['campos', 'Campos extraídos', campos.length], ['historico', 'Histórico de revisões', hist.length]].map(([k, label, n]) => (
                <button key={k} onClick={() => setTab(k)} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 7, padding: '9px 14px', fontSize: 13.5, fontWeight: 500,
                  background: 'transparent', border: 'none', borderBottom: '2px solid ' + (tab === k ? 'var(--accent)' : 'transparent'),
                  color: tab === k ? 'var(--text)' : 'var(--text-2)', marginBottom: -1,
                }}>
                  {label}
                  <span className="mono" style={{ fontSize: 11, padding: '1px 6px', borderRadius: 999, background: 'var(--subtle)', color: 'var(--text-2)' }}>{n}</span>
                </button>
              ))}
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: 18 }}>
              {tab === 'campos' && (
                <div>
                  {lowCount > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: '9px 12px', marginBottom: 14, background: 'var(--st-review-bg)', border: '1px solid ' + 'var(--st-review)' + '35', borderRadius: 'var(--r-md)' }}>
                      <span style={{ color: 'var(--st-review)' }}><Icon name="alerta" size={16} /></span>
                      <span style={{ fontSize: 12.5, color: 'var(--text)' }}><b style={{ color: 'var(--st-review)' }}>{lowCount} campos</b> com baixa confiança — destacados em âmbar para revisão.</span>
                    </div>
                  )}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    {campos.map((campo, i) => (
                      <FieldRow key={campo.campo + i} campo={campo} onChange={v => {}} onEdit={editField} />
                    ))}
                  </div>
                  <p style={{ fontSize: 11.5, color: 'var(--text-3)', marginTop: 16, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Icon name="info" size={13} /> Clique em qualquer valor para editar. Alterações são validadas e registradas na auditoria.
                  </p>
                </div>
              )}
              {tab === 'historico' && (
                <div>
                  {hist.map((item, i) => <HistoryItem key={i} item={item} last={i === hist.length - 1} />)}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Rodapé de ações */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '13px 18px', borderTop: '1px solid var(--hairline)', flexShrink: 0, background: 'var(--surface-2)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1 }}>
            <ConfidenceBar value={doc.confianca} width={120} />
            <span style={{ fontSize: 12, color: 'var(--text-2)' }}>confiança geral</span>
          </div>
          <span style={{ fontSize: 11.5, color: 'var(--text-3)', display: 'inline-flex', alignItems: 'center', gap: 12 }}>
            <span><kbd style={kbd}>A</kbd> aprovar</span>
            <span><kbd style={kbd}>R</kbd> rejeitar</span>
            <span><kbd style={kbd}>←</kbd><kbd style={kbd}>→</kbd> navegar</span>
          </span>
          <Button variant="danger" icon="x" onClick={() => setRejectOpen(true)} disabled={!!resolved}>Rejeitar</Button>
          <Button variant="primary" icon="check" onClick={aprovar} disabled={!!resolved}>Aprovar documento</Button>
        </div>
      </div>

      {/* Modal de rejeição — exige motivo */}
      {rejectOpen && (
        <div onClick={e => { e.stopPropagation(); }} style={{ position: 'fixed', inset: 0, zIndex: 260, background: 'rgba(22,32,28,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, animation: 'om-fade-in .12s ease' }}>
          <div style={{ width: 480, background: 'var(--surface)', borderRadius: 'var(--r-lg)', boxShadow: 'var(--shadow-3)', border: '1px solid var(--hairline)', padding: 22, animation: 'om-pop-in .16s ease' }}>
            <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
              <span style={{ color: 'var(--st-rejected)', background: 'var(--st-rejected-bg)', padding: 9, borderRadius: 'var(--r-md)', display: 'inline-flex', height: 'fit-content' }}><Icon name="alerta" size={20} /></span>
              <div>
                <h3 style={{ margin: 0, fontSize: 15.5, fontWeight: 600 }}>Rejeitar documento</h3>
                <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-2)' }}>A rejeição exige um motivo — fica registrado na trilha de auditoria.</p>
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 14 }}>
              {['Chave de acesso ilegível', 'Documento duplicado', 'Valores divergentes do XML', 'Documento fora do período'].map(m => (
                <button key={m} onClick={() => setRejectReason(m)} style={{
                  textAlign: 'left', padding: '9px 12px', fontSize: 13, borderRadius: 'var(--r-md)', cursor: 'pointer',
                  background: rejectReason === m ? 'var(--st-rejected-bg)' : 'var(--surface)',
                  border: '1px solid ' + (rejectReason === m ? 'var(--st-rejected)' + '50' : 'var(--hairline)'),
                  color: 'var(--text)', display: 'flex', alignItems: 'center', gap: 9,
                }}>
                  <span style={{ width: 15, height: 15, borderRadius: 999, border: '1.5px solid ' + (rejectReason === m ? 'var(--st-rejected)' : 'var(--hairline)'), display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
                    {rejectReason === m && <span style={{ width: 7, height: 7, borderRadius: 999, background: 'var(--st-rejected)' }} />}
                  </span>
                  {m}
                </button>
              ))}
            </div>
            <textarea value={rejectReason} onChange={e => setRejectReason(e.target.value)} placeholder="Motivo da rejeição…"
              style={{ width: '100%', minHeight: 70, padding: 11, fontSize: 13, fontFamily: 'var(--font-sans)', background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)', outline: 'none', resize: 'vertical', marginBottom: 16 }} />
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <Button variant="default" onClick={() => setRejectOpen(false)}>Cancelar</Button>
              <Button variant="dangerSolid" icon="x" onClick={confirmReject} disabled={!rejectReason.trim()}>Confirmar rejeição</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
const kbd = { fontFamily: 'var(--font-mono)', fontSize: 10, padding: '1px 5px', borderRadius: 4, border: '1px solid var(--hairline)', background: 'var(--surface)', color: 'var(--text-2)', margin: '0 1px' };

window.ReviewDialog = ReviewDialog;
