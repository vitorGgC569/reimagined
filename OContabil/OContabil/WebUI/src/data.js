/* ============================================================
   OContabil — Dados fictícios realistas (pt-BR)
   Exposto em window.DB
   ============================================================ */
(function () {
  'use strict';

  // ---- Helpers de formatação pt-BR ----
  function brl(n) {
    return n.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function brlFull(n) { return 'R$ ' + brl(n); }

  // ---- Clientes (empresas) ----
  const clients = [
    { id: 'c1', nome: 'Marília Comércio de Alimentos Ltda',     fantasia: 'Mercado Marília',     cnpj: '12.345.678/0001-90', uf: 'SP', municipio: 'Campinas',      regime: 'Simples Nacional', schemas: ['NF-e', 'NFS-e', 'Boleto'],            volume: 482, ativo: true },
    { id: 'c2', nome: 'Construtora Horizonte Norte S/A',         fantasia: 'Horizonte Norte',     cnpj: '08.776.443/0001-55', uf: 'MG', municipio: 'Belo Horizonte', regime: 'Lucro Real',       schemas: ['NF-e', 'CT-e', 'DARF', 'Boleto'],     volume: 367, ativo: true },
    { id: 'c3', nome: 'Drogaria São Lucas Eireli',               fantasia: 'Drogaria São Lucas',  cnpj: '23.998.112/0001-07', uf: 'SP', municipio: 'Sorocaba',       regime: 'Simples Nacional', schemas: ['NF-e', 'NFS-e'],                      volume: 311, ativo: true },
    { id: 'c4', nome: 'Transportes Vale do Aço Ltda',            fantasia: 'TransVale',           cnpj: '31.554.700/0001-21', uf: 'MG', municipio: 'Ipatinga',       regime: 'Lucro Presumido',  schemas: ['CT-e', 'NF-e', 'DARF'],               volume: 298, ativo: true },
    { id: 'c5', nome: 'Estúdio Verde Arquitetura ME',            fantasia: 'Estúdio Verde',       cnpj: '40.221.889/0001-44', uf: 'RJ', municipio: 'Niterói',        regime: 'Simples Nacional', schemas: ['NFS-e', 'Boleto', 'PIX'],             volume: 142, ativo: true },
    { id: 'c6', nome: 'Padaria e Confeitaria Pão Dourado Ltda',  fantasia: 'Pão Dourado',         cnpj: '55.013.226/0001-18', uf: 'SP', municipio: 'Jundiaí',        regime: 'Simples Nacional', schemas: ['NF-e', 'NFS-e', 'OFX'],               volume: 256, ativo: true },
    { id: 'c7', nome: 'Tecnologia Aurora Sistemas Ltda',         fantasia: 'Aurora Sistemas',     cnpj: '19.667.301/0001-72', uf: 'SP', municipio: 'São Paulo',      regime: 'Lucro Presumido',  schemas: ['NFS-e', 'DARF', 'Holerite', 'PIX'],   volume: 203, ativo: true },
    { id: 'c8', nome: 'Frigorífico Boa Carne Indústria S/A',     fantasia: 'Boa Carne',           cnpj: '02.448.915/0001-36', uf: 'GO', municipio: 'Rio Verde',       regime: 'Lucro Real',       schemas: ['NF-e', 'CT-e', 'DARF', 'Boleto'],     volume: 521, ativo: true },
    { id: 'c9', nome: 'Clínica Bem Viver Saúde Ltda',            fantasia: 'Clínica Bem Viver',   cnpj: '37.880.554/0001-09', uf: 'PR', municipio: 'Maringá',        regime: 'Simples Nacional', schemas: ['NFS-e', 'Holerite'],                  volume: 98,  ativo: true },
    { id: 'c10',nome: 'Auto Peças Velocidade Ltda',              fantasia: 'Velocidade Peças',    cnpj: '44.102.778/0001-63', uf: 'RS', municipio: 'Caxias do Sul',   regime: 'Simples Nacional', schemas: ['NF-e', 'Boleto'],                     volume: 187, ativo: false },
  ];

  // ---- Tipos de documento ----
  const docTypes = ['NF-e', 'NFS-e', 'CT-e', 'DARF', 'Boleto', 'OFX', 'Holerite', 'PIX'];

  // ---- Geração de chave de acesso (44 dígitos) ----
  function chave(seed) {
    let s = '';
    let x = seed * 2654435761 % 4294967296;
    for (let i = 0; i < 44; i++) {
      x = (x * 1103515245 + 12345) & 0x7fffffff;
      s += (x % 10);
    }
    return s;
  }
  function chaveFmt(c) { return c.replace(/(.{4})/g, '$1 ').trim(); }

  const emitentes = [
    'Distribuidora Atlântico Ltda', 'Atacadão Central de Compras S/A', 'Indústria Química Polar Ltda',
    'Comercial Três Rios ME', 'Suprimentos Gráficos União Ltda', 'Energia Luz do Sul S/A',
    'Cooperativa Agro Bandeirantes', 'Embalagens Modelo Ltda', 'Ferragens Progresso Ltda',
    'Laticínios Vale Branco S/A', 'Móveis Lar Conforto Ltda', 'Papelaria Universitária ME',
  ];

  const usuariosRev = ['Ana Beatriz Souza', 'Carlos Mendes', 'Patrícia Lima', 'Rafael Tavares', 'Juliana Prado'];

  // ---- Geração de documentos ----
  const statuses = ['aprovado', 'revisar', 'rejeitado', 'pendente', 'processando'];
  const documents = [];
  let seed = 7;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
  function pick(arr) { return arr[Math.floor(rnd() * arr.length)]; }

  const baseDate = new Date('2026-06-01T09:00:00');
  function dateOffset(daysAgo, h, m) {
    const d = new Date(baseDate);
    d.setDate(d.getDate() - daysAgo);
    d.setHours(h, m, 0, 0);
    return d;
  }
  function fmtDate(d) {
    return String(d.getDate()).padStart(2, '0') + '/' + String(d.getMonth() + 1).padStart(2, '0') + '/' + d.getFullYear();
  }
  function fmtDateTime(d) {
    return fmtDate(d) + ' ' + String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }

  let numSeq = 24817;
  for (let i = 0; i < 64; i++) {
    const cli = pick(clients.filter(c => c.ativo));
    let tipo = pick(cli.schemas.length ? cli.schemas : docTypes);
    // status com pesos
    const r = rnd();
    let status;
    if (r < 0.46) status = 'aprovado';
    else if (r < 0.68) status = 'revisar';
    else if (r < 0.78) status = 'rejeitado';
    else if (r < 0.90) status = 'pendente';
    else status = 'processando';

    // confiança correlacionada ao status
    let conf;
    if (status === 'aprovado') conf = 92 + Math.floor(rnd() * 8);
    else if (status === 'revisar') conf = 58 + Math.floor(rnd() * 22);
    else if (status === 'rejeitado') conf = 30 + Math.floor(rnd() * 28);
    else if (status === 'processando') conf = null;
    else conf = 70 + Math.floor(rnd() * 25);

    const valor = tipo === 'PIX' || tipo === 'Boleto'
      ? Math.round((120 + rnd() * 9800) * 100) / 100
      : Math.round((350 + rnd() * 48000) * 100) / 100;

    const daysAgo = Math.floor(rnd() * 9);
    const stuck = (status === 'pendente' || status === 'revisar') && daysAgo >= 3;
    const dt = dateOffset(daysAgo, 8 + Math.floor(rnd() * 9), Math.floor(rnd() * 60));

    documents.push({
      id: 'd' + (1000 + i),
      tipo,
      clienteId: cli.id,
      cliente: cli.fantasia,
      emitente: pick(emitentes),
      numero: String(numSeq--),
      serie: String(1 + Math.floor(rnd() * 3)),
      chave: chave(i + 13),
      valor,
      data: dt,
      dataStr: fmtDate(dt),
      dataHora: fmtDateTime(dt),
      status,
      confianca: conf,
      diasParado: stuck ? daysAgo : null,
      revisor: status === 'aprovado' || status === 'rejeitado' ? pick(usuariosRev) : null,
    });
  }
  // ordenar por data desc
  documents.sort((a, b) => b.data - a.data);

  // ---- Campos extraídos de exemplo (para o diálogo de revisão) ----
  function camposDe(doc) {
    if (doc.tipo === 'NF-e' || doc.tipo === 'NFS-e' || doc.tipo === 'CT-e') {
      return [
        { campo: 'Tipo de documento', valor: doc.tipo, conf: 99, tipoVal: 'texto' },
        { campo: 'Número', valor: doc.numero, conf: 98, tipoVal: 'numero' },
        { campo: 'Série', valor: doc.serie, conf: 97, tipoVal: 'numero' },
        { campo: 'Chave de acesso', valor: doc.chave, conf: doc.confianca != null ? Math.max(72, doc.confianca - 6) : 80, tipoVal: 'chave', span: 2 },
        { campo: 'CNPJ emitente', valor: '34.115.890/0001-72', conf: 95, tipoVal: 'cnpj' },
        { campo: 'Razão social emitente', valor: doc.emitente, conf: 88, tipoVal: 'texto' },
        { campo: 'CNPJ destinatário', valor: '12.345.678/0001-90', conf: 94, tipoVal: 'cnpj' },
        { campo: 'Data de emissão', valor: doc.dataStr, conf: 96, tipoVal: 'data' },
        { campo: 'CFOP', valor: '5.102', conf: doc.confianca != null && doc.confianca < 75 ? 54 : 86, tipoVal: 'numero' },
        { campo: 'Valor dos produtos', valor: brl(doc.valor * 0.86), conf: 91, tipoVal: 'moeda' },
        { campo: 'Valor do ICMS', valor: brl(doc.valor * 0.18), conf: doc.confianca != null && doc.confianca < 70 ? 49 : 83, tipoVal: 'moeda' },
        { campo: 'Valor total', valor: brl(doc.valor), conf: 93, tipoVal: 'moeda' },
      ];
    }
    if (doc.tipo === 'DARF') {
      return [
        { campo: 'Código da receita', valor: '0561', conf: 92, tipoVal: 'numero' },
        { campo: 'Período de apuração', valor: '04/2026', conf: 95, tipoVal: 'texto' },
        { campo: 'CNPJ', valor: '34.115.890/0001-72', conf: 96, tipoVal: 'cnpj' },
        { campo: 'Vencimento', valor: doc.dataStr, conf: 94, tipoVal: 'data' },
        { campo: 'Valor principal', valor: brl(doc.valor * 0.9), conf: 89, tipoVal: 'moeda' },
        { campo: 'Multa', valor: brl(doc.valor * 0.04), conf: 71, tipoVal: 'moeda' },
        { campo: 'Juros', valor: brl(doc.valor * 0.06), conf: 64, tipoVal: 'moeda' },
        { campo: 'Valor total', valor: brl(doc.valor), conf: 90, tipoVal: 'moeda' },
      ];
    }
    if (doc.tipo === 'Boleto' || doc.tipo === 'PIX') {
      return [
        { campo: 'Beneficiário', valor: doc.emitente, conf: 87, tipoVal: 'texto' },
        { campo: 'CNPJ beneficiário', valor: '34.115.890/0001-72', conf: 93, tipoVal: 'cnpj' },
        { campo: doc.tipo === 'PIX' ? 'Chave PIX' : 'Linha digitável', valor: doc.tipo === 'PIX' ? 'pagamentos@fornecedor.com.br' : '34191.79001 01043.510047 91020.150008 9 90810000' + Math.floor(doc.valor * 100), conf: 79, tipoVal: 'texto', span: 2 },
        { campo: 'Vencimento', valor: doc.dataStr, conf: 92, tipoVal: 'data' },
        { campo: 'Valor', valor: brl(doc.valor), conf: 90, tipoVal: 'moeda' },
      ];
    }
    return [
      { campo: 'Documento', valor: doc.tipo, conf: 90, tipoVal: 'texto' },
      { campo: 'Valor', valor: brl(doc.valor), conf: 85, tipoVal: 'moeda' },
      { campo: 'Data', valor: doc.dataStr, conf: 88, tipoVal: 'data' },
    ];
  }

  // ---- Histórico de revisões (auditoria) ----
  function historicoDe(doc) {
    const h = [
      { quem: 'Sistema (Motor ONNX)', acao: 'Extração automática concluída', detalhe: 'Confiança média ' + (doc.confianca ?? '—') + '%', quando: fmtDateTime(dateOffset(doc.diasParado || 1, 7, 12)), tipo: 'sistema' },
    ];
    if (doc.confianca != null && doc.confianca < 80) {
      h.push({ quem: doc.revisor || 'Ana Beatriz Souza', acao: 'Campo corrigido', detalhe: 'CFOP: "5.101" → "5.102"', quando: fmtDateTime(dateOffset(Math.max(0, (doc.diasParado || 1) - 1), 10, 33)), tipo: 'edicao' });
      h.push({ quem: doc.revisor || 'Ana Beatriz Souza', acao: 'Campo corrigido', detalhe: 'Valor do ICMS: "R$ ' + brl(doc.valor * 0.16) + '" → "R$ ' + brl(doc.valor * 0.18) + '"', quando: fmtDateTime(dateOffset(Math.max(0, (doc.diasParado || 1) - 1), 10, 35)), tipo: 'edicao' });
    }
    if (doc.status === 'aprovado') h.push({ quem: doc.revisor || 'Carlos Mendes', acao: 'Documento aprovado', detalhe: 'Pronto para exportação', quando: fmtDateTime(dateOffset(0, 11, 2)), tipo: 'aprovado' });
    if (doc.status === 'rejeitado') h.push({ quem: doc.revisor || 'Carlos Mendes', acao: 'Documento rejeitado', detalhe: 'Motivo: chave de acesso ilegível no documento original', quando: fmtDateTime(dateOffset(0, 14, 48)), tipo: 'rejeitado' });
    return h;
  }

  // ---- Usuários do sistema ----
  const users = [
    { id: 'u1', nome: 'Ana Beatriz Souza',  email: 'ana.souza@escritoriorazao.com.br',   papel: 'Administrador', ultimoLogin: '01/06/2026 08:42', status: 'ativo',     tentativas: 0, mfa: true },
    { id: 'u2', nome: 'Carlos Mendes',       email: 'carlos.mendes@escritoriorazao.com.br', papel: 'Contador',     ultimoLogin: '01/06/2026 08:15', status: 'ativo',     tentativas: 0, mfa: true },
    { id: 'u3', nome: 'Patrícia Lima',       email: 'patricia.lima@escritoriorazao.com.br', papel: 'Contador',     ultimoLogin: '31/05/2026 17:58', status: 'ativo',     tentativas: 0, mfa: false },
    { id: 'u4', nome: 'Rafael Tavares',      email: 'rafael.tavares@escritoriorazao.com.br',papel: 'Aux. Contábil',ultimoLogin: '01/06/2026 07:50', status: 'ativo',     tentativas: 2, mfa: false },
    { id: 'u5', nome: 'Juliana Prado',       email: 'juliana.prado@escritoriorazao.com.br', papel: 'Aux. Contábil',ultimoLogin: '28/05/2026 16:20', status: 'bloqueado', tentativas: 5, mfa: false },
    { id: 'u6', nome: 'Diego Antunes',       email: 'diego.antunes@escritoriorazao.com.br', papel: 'Somente leitura', ultimoLogin: '—',            status: 'convidado', tentativas: 0, mfa: false },
  ];

  // ---- Trilha de auditoria (global) ----
  const auditoria = [
    { quem: 'Ana Beatriz Souza', acao: 'Aprovou 12 documentos do cliente Mercado Marília', quando: '01/06/2026 09:14', ip: '192.168.0.14', tipo: 'aprovado' },
    { quem: 'Sistema',           acao: 'Backup automático concluído (2,4 GB)',             quando: '01/06/2026 03:00', ip: 'localhost',     tipo: 'sistema' },
    { quem: 'Carlos Mendes',     acao: 'Exportou lote para SPED Fiscal — Horizonte Norte', quando: '31/05/2026 18:22', ip: '192.168.0.21', tipo: 'export' },
    { quem: 'Juliana Prado',     acao: 'Conta bloqueada após 5 tentativas de acesso',      quando: '31/05/2026 16:41', ip: '192.168.0.39', tipo: 'seguranca' },
    { quem: 'Rafael Tavares',    acao: 'Corrigiu campo "CFOP" no documento NF-e 24.815',   quando: '31/05/2026 15:09', ip: '192.168.0.31', tipo: 'edicao' },
    { quem: 'Patrícia Lima',     acao: 'Cadastrou novo schema "NFS-e Serviços" — Aurora',  quando: '30/05/2026 11:50', ip: '192.168.0.27', tipo: 'config' },
    { quem: 'Ana Beatriz Souza', acao: 'Alterou limite de alerta de docs parados: 5 → 3',  quando: '30/05/2026 10:02', ip: '192.168.0.14', tipo: 'config' },
  ];

  // ---- KPIs e séries do dashboard ----
  const kpis = {
    processadosHoje: 148,
    naFila: 23,
    aguardandoRevisao: documents.filter(d => d.status === 'revisar').length,
    confiancaMedia: 91.4,
    docsParados: documents.filter(d => d.diasParado != null).length,
  };

  // série de confiança últimos 14 dias
  const confSeries = [];
  let cd = new Date('2026-05-19T00:00:00');
  const confVals = [88.2, 87.5, 89.1, 90.4, 88.9, 91.2, 92.0, 90.6, 89.8, 91.5, 92.3, 90.9, 91.8, 91.4];
  const volVals   = [112, 98, 134, 156, 88, 142, 167, 151, 129, 173, 188, 144, 161, 148];
  for (let i = 0; i < 14; i++) {
    const d = new Date(cd); d.setDate(d.getDate() + i);
    confSeries.push({ dia: String(d.getDate()).padStart(2, '0') + '/' + String(d.getMonth() + 1).padStart(2, '0'), conf: confVals[i], vol: volVals[i] });
  }

  // distribuição de status
  const statusDist = {
    aprovado: documents.filter(d => d.status === 'aprovado').length,
    revisar: documents.filter(d => d.status === 'revisar').length,
    rejeitado: documents.filter(d => d.status === 'rejeitado').length,
    pendente: documents.filter(d => d.status === 'pendente').length + documents.filter(d => d.status === 'processando').length,
  };

  // top clientes por volume
  const topClientes = clients.slice().sort((a, b) => b.volume - a.volume).slice(0, 6);

  // ---- Labels e cores de status ----
  const STATUS = {
    aprovado:    { label: 'Aprovado',    color: 'var(--st-approved)', bg: 'var(--st-approved-bg)' },
    revisar:     { label: 'Revisar',     color: 'var(--st-review)',   bg: 'var(--st-review-bg)' },
    rejeitado:   { label: 'Rejeitado',   color: 'var(--st-rejected)', bg: 'var(--st-rejected-bg)' },
    pendente:    { label: 'Pendente',    color: 'var(--st-pending)',  bg: 'var(--st-pending-bg)' },
    processando: { label: 'Processando', color: 'var(--st-info)',     bg: 'var(--st-info-bg)' },
  };

  window.DB = {
    clients, documents, users, auditoria, kpis, confSeries, statusDist, topClientes,
    docTypes, STATUS,
    camposDe, historicoDe, chaveFmt, brl, brlFull, fmtDate, fmtDateTime,
  };
})();
