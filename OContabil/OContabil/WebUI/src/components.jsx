/* ============================================================
   OContabil — Componentes base compartilhados
   ============================================================ */
const { useState, useEffect, useRef, useMemo, useCallback } = React;

/* ---- Badge de status ---- */
function StatusBadge({ status, size = 'md' }) {
  const s = window.DB.STATUS[status] || window.DB.STATUS.pendente;
  const pad = size === 'sm' ? '2px 7px' : '3px 9px 3px 7px';
  const dot = status === 'processando';
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: pad, borderRadius: 999, fontSize: size === 'sm' ? 11.5 : 12,
      fontWeight: 500, color: s.color, background: s.bg,
      border: '1px solid ' + s.color + '28', whiteSpace: 'nowrap', lineHeight: 1.2,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: 999, background: s.color, flexShrink: 0,
        animation: dot ? 'om-pulse 1.4s ease-in-out infinite' : 'none',
      }} />
      {s.label}
    </span>
  );
}

/* ---- Score de confiança (barra + percentual) ---- */
function confColor(v) {
  if (v == null) return 'var(--st-pending)';
  if (v >= 85) return 'var(--conf-high)';
  if (v >= 65) return 'var(--conf-mid)';
  return 'var(--conf-low)';
}
function ConfidenceBar({ value, width = 96, showLabel = true, compact = false }) {
  if (value == null) {
    return <span style={{ color: 'var(--text-3)', fontSize: 12 }} className="mono">—</span>;
  }
  const c = confColor(value);
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
      <span style={{ width, height: compact ? 5 : 6, borderRadius: 999, background: 'var(--subtle)', overflow: 'hidden', flexShrink: 0 }}>
        <span style={{ display: 'block', height: '100%', width: value + '%', background: c, borderRadius: 999, transition: 'width .5s cubic-bezier(.2,.8,.2,1)' }} />
      </span>
      {showLabel && <span className="mono" style={{ fontSize: 12, color: c, fontWeight: 500, minWidth: 30, textAlign: 'right' }}>{value}%</span>}
    </span>
  );
}

/* ---- Pílula de confiança (chip) ---- */
function ConfidencePill({ value }) {
  if (value == null) return <span className="mono" style={{ color: 'var(--text-3)' }}>—</span>;
  const c = confColor(value);
  return (
    <span className="mono" style={{
      display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11.5, fontWeight: 500,
      color: c, background: c + '15', padding: '2px 7px', borderRadius: 4, border: '1px solid ' + c + '30',
    }}>
      {value}%
    </span>
  );
}

/* ---- Botão ---- */
function Button({ children, variant = 'default', size = 'md', icon, iconRight, onClick, type = 'button', disabled, full, style = {}, title }) {
  const sizes = {
    sm: { padding: icon && !children ? '6px' : '5px 11px', fontSize: 12.5, h: 30, gap: 6 },
    md: { padding: icon && !children ? '8px' : '7px 14px', fontSize: 13.5, h: 36, gap: 7 },
    lg: { padding: '10px 18px', fontSize: 14.5, h: 44, gap: 8 },
  };
  const sz = sizes[size];
  const variants = {
    primary: { background: 'var(--accent)', color: '#fff', border: '1px solid var(--accent)' },
    default: { background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--hairline)' },
    ghost:   { background: 'transparent', color: 'var(--text-2)', border: '1px solid transparent' },
    subtle:  { background: 'var(--subtle)', color: 'var(--text)', border: '1px solid transparent' },
    danger:  { background: 'var(--surface)', color: 'var(--st-rejected)', border: '1px solid var(--st-rejected)' + '40' },
    dangerSolid: { background: 'var(--st-rejected)', color: '#fff', border: '1px solid var(--st-rejected)' },
  };
  const [hover, setHover] = useState(false);
  const v = variants[variant] || variants.default;
  const hoverStyle = hover && !disabled ? {
    primary: { background: 'var(--accent-hover)', borderColor: 'var(--accent-hover)' },
    default: { background: 'var(--surface-2)', borderColor: 'var(--text-3)' },
    ghost:   { background: 'var(--subtle)', color: 'var(--text)' },
    subtle:  { background: 'var(--hairline-2)' },
    danger:  { background: 'var(--st-rejected-bg)' },
    dangerSolid: { background: '#9c3a30' },
  }[variant] || {} : {};
  return (
    <button type={type} onClick={onClick} disabled={disabled} title={title}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: sz.gap,
        padding: sz.padding, height: sz.h, minHeight: sz.h, fontSize: sz.fontSize, fontWeight: 500,
        borderRadius: 'var(--r-md)', transition: 'all .14s ease', whiteSpace: 'nowrap',
        width: full ? '100%' : 'auto', opacity: disabled ? 0.5 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer', fontFamily: 'var(--font-sans)',
        ...v, ...hoverStyle, ...style,
      }}>
      {icon && <Icon name={icon} size={size === 'lg' ? 18 : 16} stroke={1.7} />}
      {children}
      {iconRight && <Icon name={iconRight} size={size === 'lg' ? 18 : 16} stroke={1.7} />}
    </button>
  );
}

/* ---- Controle segmentado ---- */
function Segmented({ options, value, onChange, size = 'md' }) {
  return (
    <div style={{ display: 'inline-flex', background: 'var(--subtle)', borderRadius: 'var(--r-md)', padding: 3, gap: 2 }}>
      {options.map(opt => {
        const active = opt.value === value;
        return (
          <button key={opt.value} onClick={() => onChange(opt.value)} title={opt.title}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: size === 'sm' ? '4px 9px' : '6px 12px', fontSize: size === 'sm' ? 12 : 13,
              fontWeight: 500, borderRadius: 4, border: 'none',
              background: active ? 'var(--surface)' : 'transparent',
              color: active ? 'var(--text)' : 'var(--text-2)',
              boxShadow: active ? 'var(--shadow-1)' : 'none', transition: 'all .14s ease',
            }}>
            {opt.icon && <Icon name={opt.icon} size={15} />}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

/* ---- Campo de busca ---- */
function SearchInput({ value, onChange, placeholder = 'Buscar…', width = 280, onFocus }) {
  return (
    <div style={{ position: 'relative', width }}>
      <span style={{ position: 'absolute', left: 11, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-3)', pointerEvents: 'none' }}>
        <Icon name="busca" size={16} />
      </span>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} onFocus={onFocus}
        style={{
          width: '100%', height: 36, padding: '0 12px 0 34px', fontSize: 13.5,
          background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)',
          color: 'var(--text)', outline: 'none', fontFamily: 'var(--font-sans)',
        }}
        onFocusCapture={e => e.target.style.borderColor = 'var(--accent)'}
        onBlur={e => e.target.style.borderColor = 'var(--hairline)'}
      />
    </div>
  );
}

/* ---- Select estilizado ---- */
function Select({ value, onChange, options, width, placeholder }) {
  return (
    <div style={{ position: 'relative', width: width || 'auto' }}>
      <select value={value} onChange={e => onChange(e.target.value)}
        style={{
          appearance: 'none', WebkitAppearance: 'none', width: '100%', height: 36,
          padding: '0 32px 0 12px', fontSize: 13.5, background: 'var(--surface)',
          border: '1px solid var(--hairline)', borderRadius: 'var(--r-md)', color: 'var(--text)',
          outline: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)',
        }}>
        {placeholder && <option value="">{placeholder}</option>}
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      <span style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-3)', pointerEvents: 'none' }}>
        <Icon name="chevDown" size={15} />
      </span>
    </div>
  );
}

/* ---- Card / painel ---- */
function Panel({ children, style = {}, pad = 0, className = '' }) {
  return (
    <div className={className} style={{
      background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 'var(--r-lg)',
      boxShadow: 'var(--shadow-1)', padding: pad, ...style,
    }}>{children}</div>
  );
}

/* ---- Cabeçalho de seção ---- */
function SectionHeader({ title, subtitle, action, icon }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 16, gap: 16 }}>
      <div style={{ display: 'flex', gap: 11, alignItems: 'center' }}>
        {icon && <span style={{ color: 'var(--accent)' }}><Icon name={icon} size={20} /></span>}
        <div>
          <h1 style={{ margin: 0, fontSize: 19, fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--text)' }}>{title}</h1>
          {subtitle && <p style={{ margin: '3px 0 0', fontSize: 13, color: 'var(--text-2)' }}>{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

/* ---- Selo 100% local ---- */
function LocalSeal({ variant = 'default' }) {
  if (variant === 'compact') {
    return (
      <span title="Processamento 100% local — dados nunca saem desta máquina" style={{
        display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 9px', borderRadius: 999,
        background: 'var(--accent-weak)', color: 'var(--accent-ink)', fontSize: 11.5, fontWeight: 500,
        border: '1px solid var(--accent-weak-2)',
      }}>
        <Icon name="escudo" size={13} stroke={1.8} /> 100% local
      </span>
    );
  }
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 9, padding: '8px 13px', borderRadius: 'var(--r-md)', background: 'var(--accent-weak)', border: '1px solid var(--accent-weak-2)' }}>
      <span style={{ color: 'var(--accent)' }}><Icon name="escudo" size={17} stroke={1.8} /></span>
      <span style={{ fontSize: 12.5, color: 'var(--accent-ink)', fontWeight: 500 }}>Processamento 100% local <span style={{ color: 'var(--accent)', opacity: 0.7 }}>• dados nunca saem desta máquina</span></span>
    </div>
  );
}

/* ---- Tooltip simples (hover) ---- */
function Tip({ label, children }) {
  const [show, setShow] = useState(false);
  return (
    <span style={{ position: 'relative', display: 'inline-flex' }} onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
      {children}
      {show && (
        <span style={{
          position: 'absolute', bottom: 'calc(100% + 7px)', left: '50%', transform: 'translateX(-50%)',
          background: 'var(--ink)', color: 'var(--paper)', fontSize: 11.5, padding: '5px 9px', borderRadius: 5,
          whiteSpace: 'nowrap', zIndex: 100, boxShadow: 'var(--shadow-2)', pointerEvents: 'none',
        }}>{label}</span>
      )}
    </span>
  );
}

/* ---- Avatar (iniciais) ---- */
function Avatar({ nome, size = 30 }) {
  const initials = nome.split(' ').filter(Boolean).slice(0, 2).map(w => w[0]).join('').toUpperCase();
  // hash de cor estável dentro do acento
  let h = 0; for (const ch of nome) h = (h * 31 + ch.charCodeAt(0)) & 0xffff;
  const hue = 150 + (h % 60);
  return (
    <span style={{
      width: size, height: size, borderRadius: 999, flexShrink: 0,
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      background: `oklch(0.62 0.06 ${hue})`, color: '#fff',
      fontSize: size * 0.38, fontWeight: 600, fontFamily: 'var(--font-sans)',
    }}>{initials}</span>
  );
}

/* ---- Modal genérico (overlay + painel central) ---- */
function Modal({ open, onClose, children, width = 560, labelledBy }) {
  useEffect(() => {
    if (!open) return;
    const onKey = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(22,32,28,0.42)',
      backdropFilter: 'blur(1.5px)', display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24, animation: 'om-fade-in .15s ease',
    }}>
      <div onClick={e => e.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby={labelledBy}
        style={{
          width, maxWidth: '100%', maxHeight: '90vh', overflow: 'auto', background: 'var(--surface)',
          borderRadius: 'var(--r-lg)', boxShadow: 'var(--shadow-3)', border: '1px solid var(--hairline)',
          animation: 'om-pop-in .18s cubic-bezier(.2,.8,.2,1)',
        }}>{children}</div>
    </div>
  );
}

/* ---- Toast / notificação ---- */
function Toast({ toast }) {
  if (!toast) return null;
  const colors = {
    success: { c: 'var(--st-approved)', bg: 'var(--st-approved-bg)', icon: 'check' },
    error: { c: 'var(--st-rejected)', bg: 'var(--st-rejected-bg)', icon: 'alerta' },
    info: { c: 'var(--st-info)', bg: 'var(--st-info-bg)', icon: 'info' },
  }[toast.type || 'success'];
  return (
    <div style={{
      position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 400,
      display: 'flex', alignItems: 'center', gap: 11, padding: '11px 16px', borderRadius: 'var(--r-md)',
      background: 'var(--surface)', border: '1px solid var(--hairline)', boxShadow: 'var(--shadow-3)',
      animation: 'om-pop-in .2s ease', maxWidth: 460,
    }}>
      <span style={{ color: colors.c, display: 'inline-flex', background: colors.bg, padding: 5, borderRadius: 999 }}><Icon name={colors.icon} size={15} stroke={2} /></span>
      <span style={{ fontSize: 13.5, color: 'var(--text)' }}>{toast.msg}</span>
    </div>
  );
}

/* ---- Skeleton de carregamento ---- */
function Skeleton({ width = '100%', height = 14, style = {} }) {
  return (
    <span style={{
      display: 'inline-block', width, height, borderRadius: 4,
      background: 'linear-gradient(90deg, var(--subtle) 25%, var(--hairline-2) 37%, var(--subtle) 63%)',
      backgroundSize: '800px 100%', animation: 'om-shimmer 1.4s linear infinite', ...style,
    }} />
  );
}

/* ---- Empty state útil ---- */
function EmptyState({ icon = 'documentos', title, desc, action }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '56px 24px', textAlign: 'center' }}>
      <div style={{ width: 52, height: 52, borderRadius: 'var(--r-lg)', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--subtle)', color: 'var(--text-3)', marginBottom: 16 }}>
        <Icon name={icon} size={24} />
      </div>
      <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: 'var(--text)' }}>{title}</h3>
      {desc && <p style={{ margin: '6px 0 0', fontSize: 13, color: 'var(--text-2)', maxWidth: 360 }}>{desc}</p>}
      {action && <div style={{ marginTop: 18 }}>{action}</div>}
    </div>
  );
}

/* ---- Mini sparkline / barra de gráfico (SVG simples) ---- */
function AreaChart({ data, height = 130, valueKey = 'conf', min, max, color = 'var(--accent)', labelKey = 'dia', format }) {
  const w = 100, h = 100;
  const vals = data.map(d => d[valueKey]);
  const lo = min != null ? min : Math.min(...vals) - 1;
  const hi = max != null ? max : Math.max(...vals) + 1;
  const range = hi - lo || 1;
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((d[valueKey] - lo) / range) * h;
    return [x, y];
  });
  const line = pts.map((p, i) => (i === 0 ? 'M' : 'L') + p[0].toFixed(2) + ' ' + p[1].toFixed(2)).join(' ');
  const area = line + ` L ${w} ${h} L 0 ${h} Z`;
  const gid = 'g' + valueKey;
  const [hi2, setHi2] = useState(null);
  return (
    <div style={{ position: 'relative' }}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: '100%', height, display: 'block' }}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.16" />
            <stop offset="100%" stopColor={color} stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map(g => <line key={g} x1="0" x2={w} y1={h * g} y2={h * g} stroke="var(--hairline-2)" strokeWidth="0.5" vectorEffect="non-scaling-stroke" />)}
        <path d={area} fill={`url(#${gid})`} />
        <path d={line} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={p[0]} cy={p[1]} r={hi2 === i ? 2.4 : 0} fill={color} vectorEffect="non-scaling-stroke" />
        ))}
        {data.map((d, i) => (
          <rect key={i} x={(i / data.length) * w} y="0" width={w / data.length} height={h} fill="transparent"
            onMouseEnter={() => setHi2(i)} onMouseLeave={() => setHi2(null)} style={{ cursor: 'crosshair' }} />
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 7 }}>
        {data.filter((_, i) => i % 2 === 0).map((d, i) => (
          <span key={i} className="mono" style={{ fontSize: 9.5, color: 'var(--text-3)' }}>{d[labelKey]}</span>
        ))}
      </div>
      {hi2 != null && (
        <div style={{ position: 'absolute', top: -6, left: `${(hi2 / (data.length - 1)) * 100}%`, transform: 'translate(-50%,-100%)', background: 'var(--ink)', color: 'var(--paper)', fontSize: 11, padding: '4px 8px', borderRadius: 5, whiteSpace: 'nowrap', pointerEvents: 'none', boxShadow: 'var(--shadow-2)' }}>
          <span className="mono">{format ? format(data[hi2][valueKey]) : data[hi2][valueKey]}</span>
          <span style={{ opacity: 0.6, marginLeft: 6 }} className="mono">{data[hi2][labelKey]}</span>
        </div>
      )}
    </div>
  );
}

/* ---- Botão-ícone pequeno (compartilhado entre telas) ---- */
const miniBtn = { width: 26, height: 26, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 5, color: 'var(--text-2)', cursor: 'pointer' };

Object.assign(window, {
  StatusBadge, ConfidenceBar, ConfidencePill, confColor, Button, Segmented,
  SearchInput, Select, Panel, SectionHeader, LocalSeal, Tip, Avatar, Modal,
  Toast, Skeleton, EmptyState, AreaChart, miniBtn,
});
