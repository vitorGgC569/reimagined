#!/bin/bash
set -e # Falha imediatamente se qualquer comando der erro

echo "==================================================="
echo "🛡️  OXTA INDUSTRIAL PRE-FLIGHT CHECK (PASCAL TIER)"
echo "==================================================="

# ---------------------------------------------------------
# 1. VERIFICAÇÃO DE GPU (O Coração)
# ---------------------------------------------------------
echo -n "[1/5] Checking GPU Access... "
if command -v nvidia-smi &> /dev/null; then
    # Verifica se conseguimos listar as GPUs e se não há erro
    if nvidia-smi > /dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo "✅ OK ($GPU_NAME detectada)"
    else
        echo "❌ FALHA!"
        echo "   ERRO: O driver NVIDIA não está comunicando com o container."
        echo "   SOLUÇÃO: Rode com --gpus all"
        exit 1
    fi
else
    echo "⚠️  AVISO: nvidia-smi não encontrado. Rodando em modo CPU-Only?"
fi

# ---------------------------------------------------------
# 2. VERIFICAÇÃO DE RAM (O Pulmão)
# ---------------------------------------------------------
echo -n "[2/5] Checking RAM limits... "
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))

if [ "$TOTAL_MEM_GB" -lt 10 ]; then
    echo "❌ FALHA!"
    echo "   ERRO: Detectado apenas ${TOTAL_MEM_GB}GB de RAM."
    echo "   SOLUÇÃO: Ajuste o .wslconfig para pelo menos 12GB (memory=12GB)."
    exit 1
else
    echo "✅ OK (${TOTAL_MEM_GB}GB alocados)"
fi

# ---------------------------------------------------------
# 3. VERIFICAÇÃO DE DISCO (Velocidade de I/O)
# ---------------------------------------------------------
echo -n "[3/5] Checking Disk I/O Latency... "
# Escreve 50MB e mede o tempo. Se demorar muito, é mount do Windows.
DD_RESULT=$(dd if=/dev/zero of=/tmp/iocheck bs=1M count=50 conv=fdatasync 2>&1 | tail -n 1)
SPEED_MB=$(echo "$DD_RESULT" | awk '{print $(NF-1)}')

# Nota: O parsing do dd pode variar, aqui assumimos formato "X MB/s"
# Uma verificação simples de sanidade:
if [[ "$DD_RESULT" == *"kB/s"* ]]; then
     echo "⚠️  ALERTA CRÍTICO: Disco extremamente lento (KB/s)."
     echo "   Você montou /mnt/c/? Use volumes nativos do Linux!"
     # Não abortamos, mas avisamos
elif [[ "$DD_RESULT" == *"GB/s"* ]]; then
     echo "✅ OK (NVMe Speed)"
else
     echo "✅ OK ($SPEED_MB MB/s)"
fi
rm -f /tmp/iocheck

# ---------------------------------------------------------
# 4. VERIFICAÇÃO DE REDE/DNS (Para downloads)
# ---------------------------------------------------------
echo -n "[4/5] Checking Connectivity... "
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    echo "✅ OK"
else
    echo "⚠️  AVISO: Sem internet. O download de datasets falhará."
fi

# ---------------------------------------------------------
# 5. SANIDADE DO CÓDIGO (Lint & Imports)
# ---------------------------------------------------------
echo -n "[5/5] Verifying Code Integrity... "
# Tenta importar o módulo principal para garantir que compilou
if python3 -c "import nsos_ext; print('C++ Module Loaded')" > /dev/null 2>&1; then
    echo "✅ OK (NSOS Compiled)"
else
    echo "❌ FALHA!"
    echo "   ERRO: Não foi possível importar 'nsos_ext'."
    echo "   SOLUÇÃO: Verifique o build do CMake."
    exit 1
fi

echo "==================================================="
echo "🚀 ALL SYSTEMS GO. STARTING ENGINE."
echo "==================================================="

# Executa o comando passado para o docker (ex: /bin/bash ou python train.py)
exec "$@"
