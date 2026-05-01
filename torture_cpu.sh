#!/bin/bash
echo "🔥 INICIANDO TESTE DE DETERMINISMO INDUSTRIAL (CPU) 🔥"
echo "Alvo: 50 execuções idênticas bit-a-bit."

# 1. Limpar artefatos anteriores
rm -f run_*.log
mkdir -p logs

# 2. Executar 50 vezes em loop
for i in {1..50}
do
   # Executa o treino e salva APENAS o hash/soma de verificação da saída
   # Supondo que seu script python printe "FINAL_HASH: <valor>" no final
   echo -ne "Execução #$i... "

   # Força 1 thread para validar lógica pura primeiro
   export OMP_NUM_THREADS=1

   # Usamos o generate_golden.py pois ele é mais rápido e focado em determinismo
   python3 OXN/tests/generate_golden.py > "logs/run_$i.log" 2>&1

   # Verifica se houve Crash (Segfault)
   if [ $? -ne 0 ]; then
       echo "❌ CRASH DETECTADO (Segfault/Abort) na run #$i"
       echo "Verifique logs/run_$i.log imediatamente."
       exit 1
   fi

   echo "✅ Concluído."
done

# 3. Comparação Forense
echo "🔍 Comparando saídas..."
# Extract just the hashes for comparison
FIRST_HASH=$(grep "Output Hash" logs/run_1.log | awk '{print $3}')
DIFF_COUNT=0

if [ -z "$FIRST_HASH" ]; then
    echo "❌ FALHA: Não encontrou hash no log 1. Verifique se o script rodou."
    cat logs/run_1.log
    exit 1
fi

for i in {2..50}
do
   CURRENT_HASH=$(grep "Output Hash" logs/run_$i.log | awk '{print $3}')
   if [ "$FIRST_HASH" != "$CURRENT_HASH" ]; then
       echo "❌ DIVERGÊNCIA NA RUN #$i!"
       echo "Run #1 Hash: $FIRST_HASH"
       echo "Run #$i Hash: $CURRENT_HASH"
       diff logs/run_1.log logs/run_$i.log | head -n 20
       exit 1
   fi
done

echo "🏆 SUCESSO ABSOLUTO: O Oxta é Determinístico na CPU."
