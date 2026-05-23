/* ============================================================
   ContaDoc AI — Mock Data for Frontend Development
   Realistic Brazilian accounting document data
   ============================================================ */

const MockData = {
  // ── Dashboard Stats ───────────────────────────────────────
  stats: {
    processedToday: 342,
    processedTodayTrend: '+18%',
    awaitingValidation: 5,
    awaitingValidationTrend: '-12%',
    readingErrors: 2,
    totalClients: 47,
    totalDocumentsMonth: 8743,
    avgProcessingTime: '1.8s',
    accuracyRate: 96.4,
  },

  // ── Volume Chart (last 14 days) ───────────────────────────
  volumeChart: [
    { day: '23/02', count: 287 },
    { day: '24/02', count: 312 },
    { day: '25/02', count: 198 },
    { day: '26/02', count: 456 },
    { day: '27/02', count: 389 },
    { day: '28/02', count: 401 },
    { day: '01/03', count: 523 },
    { day: '02/03', count: 478 },
    { day: '03/03', count: 267 },
    { day: '04/03', count: 356 },
    { day: '05/03', count: 512 },
    { day: '06/03', count: 445 },
    { day: '07/03', count: 389 },
    { day: '08/03', count: 342 },
  ],

  // ── Clients ───────────────────────────────────────────────
  clients: [
    {
      id: 1,
      name: 'Construtora Horizonte Ltda',
      cnpj: '12.345.678/0001-90',
      color: '#6366f1',
      docsMonth: 450,
      docsValidated: 432,
      pendingDocs: 18,
    },
    {
      id: 2,
      name: 'Restaurante Sabor & Arte ME',
      cnpj: '23.456.789/0001-01',
      color: '#8b5cf6',
      docsMonth: 380,
      docsValidated: 380,
      pendingDocs: 0,
    },
    {
      id: 3,
      name: 'Tech Solutions Informática Ltda',
      cnpj: '34.567.890/0001-12',
      color: '#06b6d4',
      docsMonth: 612,
      docsValidated: 590,
      pendingDocs: 22,
    },
    {
      id: 4,
      name: 'Farmácia Popular São João',
      cnpj: '45.678.901/0001-23',
      color: '#10b981',
      docsMonth: 890,
      docsValidated: 878,
      pendingDocs: 12,
    },
    {
      id: 5,
      name: 'Auto Peças Nacional Eireli',
      cnpj: '56.789.012/0001-34',
      color: '#f59e0b',
      docsMonth: 234,
      docsValidated: 234,
      pendingDocs: 0,
    },
    {
      id: 6,
      name: 'Clínica Odontológica Sorrisos',
      cnpj: '67.890.123/0001-45',
      color: '#ef4444',
      docsMonth: 156,
      docsValidated: 150,
      pendingDocs: 6,
    },
    {
      id: 7,
      name: 'Supermercado Bom Preço Ltda',
      cnpj: '78.901.234/0001-56',
      color: '#3b82f6',
      docsMonth: 1240,
      docsValidated: 1200,
      pendingDocs: 40,
    },
    {
      id: 8,
      name: 'Academia Corpo em Forma ME',
      cnpj: '89.012.345/0001-67',
      color: '#ec4899',
      docsMonth: 98,
      docsValidated: 98,
      pendingDocs: 0,
    },
  ],

  // ── Documents awaiting validation ──────────────────────────
  validationQueue: [
    {
      id: 'DOC-001',
      filename: 'NF_2026_001234.pdf',
      client: 'Construtora Horizonte Ltda',
      clientId: 1,
      type: 'Nota Fiscal',
      uploadedAt: '2026-03-08T18:30:00',
      status: 'ready_for_review',
      extracted: {
        cnpj_emissor: { value: '12.345.678/0001-90', confidence: 0.97, span: [45, 63] },
        razao_social: { value: 'Construtora Horizonte Ltda', confidence: 0.95, span: [12, 38] },
        valor_total: { value: 'R$ 15.750,00', confidence: 0.92, span: [120, 132] },
        data_emissao: { value: '08/03/2026', confidence: 0.98, span: [78, 88] },
        numero_nf: { value: '001234', confidence: 0.99, span: [95, 101] },
        descricao: { value: 'Serviços de construção civil - Etapa 3', confidence: 0.85, span: [140, 178] },
      },
      ocrText: `NOTA FISCAL DE SERVIÇOS ELETRÔNICA - NFS-e

Razão Social: Construtora Horizonte Ltda
CNPJ: 12.345.678/0001-90
Inscrição Municipal: 1234567

Data de Emissão: 08/03/2026
Número da Nota: 001234

DISCRIMINAÇÃO DOS SERVIÇOS:

Serviços de construção civil - Etapa 3
Conforme contrato nº 456/2025

VALOR TOTAL: R$ 15.750,00

ISS Retido: R$ 787,50 (5%)
Valor Líquido: R$ 14.962,50

Tomador do Serviço:
Empresa ABC Empreendimentos S.A.
CNPJ: 98.765.432/0001-10

Esta NFS-e foi emitida conforme Lei Complementar 116/2003.`
    },
    {
      id: 'DOC-002',
      filename: 'boleto_energia_mar2026.pdf',
      client: 'Restaurante Sabor & Arte ME',
      clientId: 2,
      type: 'Boleto',
      uploadedAt: '2026-03-08T17:45:00',
      status: 'ready_for_review',
      extracted: {
        cnpj_emissor: { value: '33.000.167/0001-01', confidence: 0.94, span: [30, 48] },
        razao_social: { value: 'CEMIG Distribuição S.A.', confidence: 0.91, span: [10, 32] },
        valor_total: { value: 'R$ 2.340,67', confidence: 0.88, span: [95, 106] },
        data_vencimento: { value: '15/03/2026', confidence: 0.96, span: [65, 75] },
        codigo_barras: { value: '23793.38128 60000.000003 00000.000408 1 84340000234067', confidence: 0.72, span: [180, 232] },
      },
      ocrText: `CONTA DE ENERGIA ELÉTRICA

CEMIG Distribuição S.A.
CNPJ: 33.000.167/0001-01

Unidade Consumidora: 0012345678
Classe: Comercial - Trifásico

Mês Referência: MARÇO/2026
Data de Vencimento: 15/03/2026

Consumo: 1.890 kWh
Demanda: 45 kW

VALOR TOTAL: R$ 2.340,67

Código de Barras:
23793.38128 60000.000003 00000.000408 1 84340000234067

Chave de Acesso PIX para pagamento disponível.

Restaurante Sabor & Arte ME
CNPJ: 23.456.789/0001-01
Endereço: Rua das Flores, 123 - Centro`
    },
    {
      id: 'DOC-003',
      filename: 'comprovante_pix_08mar.jpg',
      client: 'Tech Solutions Informática Ltda',
      clientId: 3,
      type: 'Comprovante PIX',
      uploadedAt: '2026-03-08T16:20:00',
      status: 'ready_for_review',
      extracted: {
        cnpj_pagador: { value: '34.567.890/0001-12', confidence: 0.89, span: [20, 38] },
        nome_pagador: { value: 'Tech Solutions Informática Ltda', confidence: 0.93, span: [5, 36] },
        valor: { value: 'R$ 4.500,00', confidence: 0.96, span: [80, 91] },
        data_pagamento: { value: '08/03/2026', confidence: 0.98, span: [55, 65] },
        chave_pix: { value: 'financeiro@fornecedor.com.br', confidence: 0.78, span: [100, 128] },
        nome_recebedor: { value: 'Distribuidora Digital Express', confidence: 0.82, span: [130, 159] },
      },
      ocrText: `COMPROVANTE DE TRANSFERÊNCIA PIX

Pagador: Tech Solutions Informática Ltda
CNPJ: 34.567.890/0001-12
Banco: 001 - Banco do Brasil

Data/Hora: 08/03/2026 14:35:22

Valor da Transferência: R$ 4.500,00

Tipo: PIX
Chave: financeiro@fornecedor.com.br

Recebedor: Distribuidora Digital Express
CNPJ: 11.222.333/0001-44
Banco: 341 - Itaú

ID da Transação: E00000000202603081435ABCDEF1234
Autenticação: 9F8E7D6C5B4A`
    },
    {
      id: 'DOC-004',
      filename: 'DARF_IRPJ_fev2026.pdf',
      client: 'Farmácia Popular São João',
      clientId: 4,
      type: 'Guia DARF',
      uploadedAt: '2026-03-08T15:10:00',
      status: 'ready_for_review',
      extracted: {
        cnpj: { value: '45.678.901/0001-23', confidence: 0.97, span: [25, 43] },
        razao_social: { value: 'Farmácia Popular São João', confidence: 0.94, span: [5, 30] },
        codigo_receita: { value: '2089', confidence: 0.91, span: [60, 64] },
        periodo_apuracao: { value: '28/02/2026', confidence: 0.93, span: [70, 80] },
        valor_principal: { value: 'R$ 8.234,50', confidence: 0.95, span: [110, 121] },
        data_vencimento: { value: '31/03/2026', confidence: 0.97, span: [85, 95] },
      },
      ocrText: `DOCUMENTO DE ARRECADAÇÃO DE RECEITAS FEDERAIS - DARF

Farmácia Popular São João
CNPJ: 45.678.901/0001-23

Código da Receita: 2089 - IRPJ
Período de Apuração: 28/02/2026
Data de Vencimento: 31/03/2026

Número de Referência: 0000

VALOR DO PRINCIPAL: R$ 8.234,50
MULTA: R$ 0,00
JUROS/ENCARGOS: R$ 0,00
VALOR TOTAL: R$ 8.234,50

Autenticação Bancária:
123.456.789-0 08/03/2026 AG: 1234`
    },
    {
      id: 'DOC-005',
      filename: 'extrato_banco_fev2026.pdf',
      client: 'Supermercado Bom Preço Ltda',
      clientId: 7,
      type: 'Extrato Bancário',
      uploadedAt: '2026-03-08T14:00:00',
      status: 'ready_for_review',
      extracted: {
        cnpj: { value: '78.901.234/0001-56', confidence: 0.96, span: [15, 33] },
        banco: { value: 'Bradesco S.A.', confidence: 0.90, span: [5, 18] },
        agencia: { value: '1234-5', confidence: 0.88, span: [40, 46] },
        conta: { value: '56789-0', confidence: 0.92, span: [50, 57] },
        saldo_anterior: { value: 'R$ 45.230,12', confidence: 0.85, span: [70, 82] },
        saldo_final: { value: 'R$ 52.780,45', confidence: 0.87, span: [200, 212] },
      },
      ocrText: `EXTRATO DE CONTA CORRENTE

Bradesco S.A.
CNPJ: 78.901.234/0001-56

Agência: 1234-5
Conta Corrente: 56789-0

Período: 01/02/2026 a 28/02/2026

SALDO ANTERIOR: R$ 45.230,12

DATA       HISTÓRICO                    VALOR
01/02  Depósito TEF              +R$ 12.500,00
03/02  Pagto Fornecedor          -R$ 3.450,00
05/02  PIX Recebido              +R$ 8.900,00
10/02  Folha Pagamento          -R$ 15.600,00
15/02  Tarifa Bancária            -R$ 89,67
20/02  Depósito TEF              +R$ 18.300,00
25/02  Pagto Impostos            -R$ 5.230,00
28/02  PIX Enviado               -R$ 7.800,00

SALDO FINAL: R$ 52.780,45

Total de créditos: R$ 39.700,00
Total de débitos: R$ 32.169,67`
    },
  ],

  // ── Recent activity for dashboard ──────────────────────────
  recentActivity: [
    { time: '18:30', action: 'NF processada', client: 'Construtora Horizonte', status: 'success', doc: 'NF_2026_001234.pdf' },
    { time: '18:28', action: 'Boleto extraído', client: 'Restaurante Sabor & Arte', status: 'warning', doc: 'boleto_energia_mar2026.pdf' },
    { time: '18:15', action: 'PIX validado', client: 'Tech Solutions', status: 'success', doc: 'comprovante_pix_08mar.jpg' },
    { time: '17:50', action: 'DARF processado', client: 'Farmácia Popular', status: 'success', doc: 'DARF_IRPJ_fev2026.pdf' },
    { time: '17:30', action: 'Erro OCR', client: 'Clínica Sorrisos', status: 'error', doc: 'recibo_torto_scan.jpg' },
    { time: '17:15', action: 'Extrato extraído', client: 'Supermercado Bom Preço', status: 'success', doc: 'extrato_banco_fev2026.pdf' },
    { time: '16:45', action: 'Lote processado', client: 'Auto Peças Nacional', status: 'success', doc: '12 documentos' },
    { time: '16:20', action: 'NF validada', client: 'Academia Corpo em Forma', status: 'success', doc: 'NF_servico_0089.pdf' },
  ],

  // ── Processing queue (upload page) ─────────────────────────
  processingQueue: [
    { id: 'Q-001', filename: 'lote_nfs_marco.zip', size: '12.4 MB', filesCount: 25, status: 'processing', progress: 68 },
    { id: 'Q-002', filename: 'NF_servico_0145.pdf', size: '340 KB', filesCount: 1, status: 'processing', progress: 90 },
    { id: 'Q-003', filename: 'fotos_whatsapp_recibos.zip', size: '8.2 MB', filesCount: 15, status: 'queued', progress: 0 },
    { id: 'Q-004', filename: 'extrato_itau_jan.pdf', size: '1.1 MB', filesCount: 1, status: 'completed', progress: 100 },
    { id: 'Q-005', filename: 'guias_impostos_fev.pdf', size: '2.3 MB', filesCount: 8, status: 'completed', progress: 100 },
  ],

  // ── GLiNER 2 Schema templates ──────────────────────────────
  extractionSchemas: {
    nota_fiscal: {
      label: 'Nota Fiscal',
      fields: [
        'cnpj_emissor::str::CNPJ da empresa emissora',
        'razao_social::str::Nome/Razão Social do emissor',
        'numero_nf::str::Número da nota fiscal',
        'data_emissao::str::Data de emissão (DD/MM/AAAA)',
        'valor_total::str::Valor total da nota fiscal',
        'descricao::str::Descrição dos serviços ou produtos',
      ],
    },
    boleto: {
      label: 'Boleto Bancário',
      fields: [
        'cnpj_emissor::str::CNPJ do emissor do boleto',
        'razao_social::str::Nome/Razão Social',
        'valor_total::str::Valor do boleto',
        'data_vencimento::str::Data de vencimento',
        'codigo_barras::str::Código de barras numérico',
      ],
    },
    comprovante_pix: {
      label: 'Comprovante PIX',
      fields: [
        'cnpj_pagador::str::CNPJ de quem pagou',
        'nome_pagador::str::Nome de quem pagou',
        'valor::str::Valor transferido',
        'data_pagamento::str::Data e hora do pagamento',
        'chave_pix::str::Chave PIX do recebedor',
        'nome_recebedor::str::Nome de quem recebeu',
      ],
    },
    darf: {
      label: 'Guia DARF',
      fields: [
        'cnpj::str::CNPJ do contribuinte',
        'razao_social::str::Nome/Razão Social',
        'codigo_receita::str::Código da receita federal',
        'periodo_apuracao::str::Período de apuração',
        'valor_principal::str::Valor principal do imposto',
        'data_vencimento::str::Data de vencimento',
      ],
    },
  },
};
