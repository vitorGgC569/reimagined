/* ============================================================
   OContabil — Ícones de linha (stroke, 24x24, traço 1.6)
   Uso: <Icon name="documentos" size={18} />
   ============================================================ */

const ICON_PATHS = {
  painel:      '<path d="M3 13h8V3H3v10Zm0 8h8v-6H3v6Zm10 0h8V11h-8v10Zm0-18v6h8V3h-8Z"/>',
  documentos:  '<path d="M14 3v5h5"/><path d="M7 3h8l5 5v13H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"/><path d="M9 13h6M9 17h6"/>',
  clientes:    '<path d="M3 21V9l9-6 9 6v12"/><path d="M9 21v-6h6v6"/><path d="M9 11h.01M15 11h.01"/>',
  schemas:     '<path d="M8 3H6a2 2 0 0 0-2 2v3a2 2 0 0 1-2 2 2 2 0 0 1 2 2v3a2 2 0 0 0 2 2h2"/><path d="M16 3h2a2 2 0 0 1 2 2v3a2 2 0 0 0 2 2 2 2 0 0 0-2 2v3a2 2 0 0 1-2 2h-2"/>',
  exportacoes: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  usuarios:    '<circle cx="9" cy="8" r="3.2"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0"/><path d="M16 6.2a3 3 0 0 1 0 5.6"/><path d="M17.5 14.4A5.5 5.5 0 0 1 20.5 19.5"/>',
  config:      '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2V21a2 2 0 1 1-4 0v-.1A1.7 1.7 0 0 0 7 19.3a1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 2.6 14H2.5a2 2 0 1 1 0-4h.1A1.7 1.7 0 0 0 4.3 7a1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 9 2.6h.1A1.7 1.7 0 0 0 10.7 1a2 2 0 1 1 4 0v.1A1.7 1.7 0 0 0 17 2.7a1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9v.1A1.7 1.7 0 0 0 23 10.7h0a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1Z"/>',
  busca:       '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
  comando:     '<path d="M9 6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6Z"/>',
  filtro:      '<path d="M3 5h18l-7 8v6l-4-2v-4L3 5Z"/>',
  mais:        '<path d="M12 5v14M5 12h14"/>',
  check:       '<path d="m4 12 5 5 11-11"/>',
  x:           '<path d="M6 6l12 12M18 6 6 18"/>',
  alerta:      '<path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v5M12 18h.01"/>',
  escudo:      '<path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6l-7-3Z"/><path d="m9 12 2 2 4-4"/>',
  cadeado:     '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
  sol:         '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  lua:         '<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z"/>',
  linhas:      '<path d="M4 6h16M4 12h16M4 18h16"/>',
  linhasComp:  '<path d="M4 5h16M4 9h16M4 13h16M4 17h16M4 21h16"/>',
  chevDown:    '<path d="m6 9 6 6 6-6"/>',
  chevRight:   '<path d="m9 6 6 6-6 6"/>',
  chevLeft:    '<path d="m15 6-6 6 6 6"/>',
  setaDir:     '<path d="M5 12h14M13 6l6 6-6 6"/>',
  lapis:       '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5Z"/>',
  relogio:     '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  copiar:      '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>',
  olho:        '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
  download:    '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
  planilha:    '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>',
  arquivo:     '<path d="M14 3v5h5"/><path d="M7 3h8l5 5v13H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"/>',
  sair:        '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/>',
  ponto:       '<circle cx="12" cy="12" r="5"/>',
  raio:        '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/>',
  copiloto:    '<path d="M12 3v3M5.5 6.5l2 2M18.5 6.5l-2 2"/><rect x="6" y="9" width="12" height="9" rx="2.5"/><path d="M10 13h.01M14 13h.01"/><path d="M9 18l-1.5 3M15 18l1.5 3"/>',
  enviar:      '<path d="M4 12 20 4l-6 16-3-7-7-1Z"/>',
  ferramenta:  '<path d="M14.5 5.5a3.5 3.5 0 0 0-4.9 4.2L3 16.3 4.7 18l6.6-6.6a3.5 3.5 0 0 0 4.2-4.9l-2.1 2.1-2-2 2.1-2.1Z"/>',
  cifra:       '<path d="M12 2v20M17 6c0-2-2-3-5-3s-5 1-5 3.5S9 10 12 10.5s5 1 5 3.5-2 3.5-5 3.5-5-1-5-3"/>',
  calendario:  '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 9h18M8 3v4M16 3v4"/>',
  pasta:       '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
  caminhao:    '<path d="M3 7h11v8H3zM14 10h4l3 3v2h-7z"/><circle cx="7" cy="18" r="1.6"/><circle cx="17.5" cy="18" r="1.6"/>',
  recibo:      '<path d="M5 3v18l2-1.4L9 21l2-1.4L13 21l2-1.4L17 21l2-1.4V3l-2 1.4L15 3l-2 1.4L11 3 9 4.4 7 3 5 4.4Z"/><path d="M8 8h8M8 12h8"/>',
  banco:       '<path d="M3 10 12 4l9 6"/><path d="M5 10v8M9 10v8M15 10v8M19 10v8M3 21h18"/>',
  pessoas:     '<circle cx="9" cy="8" r="3.2"/><path d="M3.5 20a5.5 5.5 0 0 1 11 0"/>',
  info:        '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  spinner:     '<path d="M12 3a9 9 0 1 0 9 9" />',
  upload:      '<path d="M12 21V9"/><path d="m7 14 5-5 5 5"/><path d="M5 3h14"/>',
  estrela:     '<path d="M12 3l2.6 5.6 6 .8-4.4 4.2 1.1 6L12 16.8 6.7 19.6l1.1-6L3.4 9.4l6-.8L12 3Z"/>',
  grade:       '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
};

function Icon({ name, size = 18, stroke = 1.6, className = '', style = {}, fill = 'none' }) {
  const d = ICON_PATHS[name] || ICON_PATHS.ponto;
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      fill={fill} stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round"
      className={className} style={{ flexShrink: 0, ...style }}
      dangerouslySetInnerHTML={{ __html: ICON_PATHS[name] || ICON_PATHS.ponto }}
      aria-hidden="true"
    />
  );
}

// ícone por tipo de documento
const DOC_ICON = {
  'NF-e': 'arquivo', 'NFS-e': 'recibo', 'CT-e': 'caminhao',
  'DARF': 'cifra', 'Boleto': 'banco', 'OFX': 'banco',
  'Holerite': 'pessoas', 'PIX': 'raio',
};

window.Icon = Icon;
window.ICON_PATHS = ICON_PATHS;
window.DOC_ICON = DOC_ICON;
