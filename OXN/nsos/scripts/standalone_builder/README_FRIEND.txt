OxtaTrainer v1
==============

Obrigado por rodar o treino. Esse arquivo tem tudo que voce precisa.

PRE-REQUISITOS NO PC
---------------------
- Windows 10 ou 11, 64-bit
- NVIDIA driver versao 535 ou mais nova
- CUDA Toolkit 12.x instalado (https://developer.nvidia.com/cuda-downloads)
- ~80GB livre em disco (treino gera checkpoints intermediarios)
- GPU NVIDIA RTX 20-series ou mais nova com 11GB+ VRAM

Pra verificar se ta tudo certo, abre Prompt de Comando e roda:

    nvidia-smi

Se aparecer o nome da sua GPU e a versao do driver, ta ok.


COMO USAR
---------
1. Descompacta esse arquivo numa pasta com pelo menos 80GB livres.
   (Recomendo: D:\OxtaTrainer\  ou  C:\OxtaTrainer\)

2. Abre a pasta extraida. Voce deve ver:

       OxtaTrainer.exe
       data\
       output\        (vazia inicialmente)
       README.txt     (este arquivo)
       _internal\     (arquivos de sistema, nao mexe)

3. Da duplo clique em OxtaTrainer.exe.

   Vai abrir uma janela preta (Prompt de Comando). Isso e normal.
   NAO feche essa janela. Se fechar, o treino para.

4. Aguarda. O treino vai durar cerca de 6 dias rodando 24/7,
   ou 9-10 dias se voce usar o computador pra outras coisas
   durante o dia.

   Voce pode pausar a qualquer momento (Ctrl+C ou fechar a janela)
   sem perder progresso — o programa salva checkpoints automaticos
   a cada 25 minutos. Pra retomar, e so dar duplo clique em
   OxtaTrainer.exe de novo: ele detecta o ultimo checkpoint
   e continua de onde parou.


O QUE ACOMPANHAR DURANTE O TREINO
----------------------------------
A janela preta vai mostrar linhas como:

    [15:42:18]   step= 1500/50000  loss=4.231  step_t=2.34s  eta=86.2h

- "step" = quantos passos ja foram
- "loss" = qualidade atual (deve descer de ~10 ate ~3-4)
- "eta"  = quanto tempo falta pra terminar

Se voce ver loss DESCENDO ao longo do tempo, ta tudo certo.

Pode ver tambem em output\logs\session.log se quiser revisar depois.


SE DER ERRO
-----------
Algumas situacoes comuns:

* "CUDA out of memory"
  Significa que outro programa esta usando a GPU. Fecha jogos,
  navegadores com aceleracao GPU, OBS, etc. e tenta de novo.

* "GPU NVIDIA nao encontrada"
  Driver NVIDIA nao instalado ou desatualizado. Instala/atualiza.

* O programa fechou sozinho
  Geralmente queda de luz ou crash temporario. Roda OxtaTrainer.exe
  de novo — ele retoma do ultimo checkpoint salvo. Voce perde no
  maximo 25 minutos.

* GPU muito quente (passa 85 graus C constante)
  Risco de throttling. Abre o gabinete, limpa poeira, verifica
  ventoinhas. Em ultimo caso, pausa 30 minutos pra esfriar.


QUANDO TERMINAR
---------------
Quando voce ver na janela:

    ============================================================
     Treino completo em X.Xh
    ============================================================

Va na pasta output\final_pack\

Voce vai encontrar os arquivos do modelo final. Compacta a pasta
inteira num arquivo .zip e manda pro operador (Vitor) pelo canal
combinado (Drive, WeTransfer, etc).

A pasta output\checkpoints\ pode ser apagada — sao apenas estados
intermediarios. O operador so precisa do final_pack\.


CONSUMO DE ENERGIA
------------------
O programa limita a GPU a 220W (de 250W) pra reduzir temperatura.
Durante 6 dias de treino, isso da cerca de 32 kWh, ou aproximadamente
R$ 25 a R$ 30 em conta de luz. Combina com o Vitor pra ressarcir.


DUVIDAS
-------
Chama o Vitor pelo canal habitual. Manda print da janela ou cola
o texto do erro pra ele.

Obrigado de novo. Esse modelo nao existiria sem voce.
