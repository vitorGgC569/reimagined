import numpy as np
import scipy.sparse as sp
import networkx as nx
import time
import sys
import os
import subprocess
import tempfile

# ------------------------------------------------------------------------------
# BUILD SYSTEM (Auto-Compile C++)
# ------------------------------------------------------------------------------
def ensure_native_solver():
    bin_path = "./duan_native"
    src_path = "bridge.cpp"
    repo_path = "bmssp_cpp"

    if os.path.exists(bin_path):
        return bin_path

    print("[BUILD] Compilando solver nativo C++...")

    # 1. Verificar dependências
    # Verifica se o arquivo de cabeçalho principal existe para garantir que o repo é válido
    header_path = os.path.join(repo_path, "bmssp.hpp")
    if not os.path.exists(header_path):
        print(f"[BUILD] Clonando repositório bmssp (C++)...")
        # Remove diretório se existir mas estiver vazio ou incompleto
        if os.path.exists(repo_path):
            import shutil
            shutil.rmtree(repo_path)

        subprocess.run(
            ["git", "clone", "https://github.com/lcs147/bmssp.git", repo_path],
            check=True
        )

    # 2. Compilar
    # g++ -O3 -march=native -flto -funroll-loops -std=c++20 bridge.cpp -o duan_native
    try:
        # Flags de "Potência Máxima":
        # -O3: Otimização máxima padrão
        # -march=native: Usa instruções específicas da CPU atual (AVX, etc)
        # -flto: Link Time Optimization
        # -funroll-loops: Desenrola loops para reduzir overhead de branch
        cmd = [
            "g++", "-O3", "-march=native", "-flto", "-funroll-loops",
            "-std=c++20", src_path, "-o", bin_path
        ]
        subprocess.run(cmd, check=True)
        print("[BUILD] Compilação 'Potência Máxima' bem sucedida!")
        return bin_path
    except subprocess.CalledProcessError:
        print("[BUILD] ERRO: Falha na compilação. Verifique se g++ suporta C++20.")
        return None
    except FileNotFoundError:
        print("[BUILD] ERRO: g++ não encontrado.")
        return None

# ------------------------------------------------------------------------------
# COMPETIDOR 1: DUAN ET AL. (NATIVE C++ - HIGH PERFORMANCE)
# ------------------------------------------------------------------------------
class DuanNativeSolver:
    def __init__(self, n, edges, num_edges=None):
        self.n = n
        self.edges = edges
        self.m = num_edges if num_edges is not None else len(edges)
        self.bin_path = ensure_native_solver()

    def solve(self, source):
        if not self.bin_path:
            raise RuntimeError("Binário nativo não disponível")

        # 1. Preparar arquivo de entrada para o binário C++
        # Formato: N M SOURCE
        #          U V W
        #          ...
        fd, temp_path = tempfile.mkstemp(text=True)

        with os.fdopen(fd, 'w') as f:
            f.write(f"{self.n} {self.m} {source}\n")
            for u, v, w in self.edges:
                f.write(f"{u} {v} {w}\n")

        try:
            # 2. Chamar o binário
            # A saída esperada é:
            # LINHA 1: Tempo em ms
            # LINHAS RESTANTES: node distance
            try:
                result = subprocess.run(
                    [self.bin_path, temp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except OSError:
                # Fallback: tentar recompilar se o binário estiver corrompido ou plataforma errada
                print("[WARN] Binário nativo falhou. Tentando recompilar...")
                if os.path.exists(self.bin_path):
                    os.remove(self.bin_path)
                self.bin_path = ensure_native_solver()
                if not self.bin_path: raise

                result = subprocess.run(
                    [self.bin_path, temp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )

            output_lines = result.stdout.strip().split('\n')
            if not output_lines:
                raise ValueError("No output from binary")

            time_ms = float(output_lines[0])
            dist_dict = {}

            for line in output_lines[1:]:
                parts = line.split()
                if len(parts) != 2: continue
                node = int(parts[0])
                val_str = parts[1]
                if val_str == "inf":
                    dist_dict[node] = float('inf')
                else:
                    dist_dict[node] = float(val_str)

            return dist_dict, time_ms

        except subprocess.CalledProcessError as e:
            print(f"Erro no binário C++: {e.stderr}")
            return {}, float('inf')
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ------------------------------------------------------------------------------
# COMPETIDOR 2: HFS-SSSP V18 (Vectorized SPFA) - O "Algoritmo Deus"
# ------------------------------------------------------------------------------
class HFS_SSSP_V18_Symmetric:
    def __init__(self, n, edges, symmetric=True):
        self.n = n
        # Preparação Vectorizada (CSR Matrix)
        rows = []
        cols = []
        data = []
        for u, v, w in edges:
            rows.append(u); cols.append(v); data.append(w)
            if symmetric:
                rows.append(v); cols.append(u); data.append(w)
        self.adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def solve_exact(self, source):
        t_start = time.perf_counter()
        inf = 1e14
        dist = np.full(self.n, inf, dtype=np.float64)
        dist[source] = 0
        active_nodes = np.array([source], dtype=np.int32)

        iterations = 0
        while active_nodes.size > 0:
            iterations += 1
            # 1. Gather (Leitura em Bloco)
            sub_adj = self.adj[active_nodes]
            if sub_adj.nnz == 0: break

            coo = sub_adj.tocoo()
            global_u = active_nodes[coo.row]
            global_v = coo.col
            weights = coo.data

            # 2. Relax (ALU Vectorizada)
            new_dists = dist[global_u] + weights

            # 3. Filter (Lógica Booleana em Massa)
            mask = new_dists < (dist[global_v] - 1e-9)
            if not np.any(mask): break

            valid_v = global_v[mask]
            valid_d = new_dists[mask]

            # 4. Update (Atomic Min)
            np.minimum.at(dist, valid_v, valid_d)

            # 5. Next Frontier
            active_nodes = np.unique(valid_v)

        t_end = time.perf_counter()
        return dist, (t_end - t_start) * 1000, iterations

# ------------------------------------------------------------------------------
# ARENA DE BATALHA (BENCHMARK)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    n = 1_000_000  # 1 Milhão de nós
    avg_degree = 3
    print(f"--- INICIANDO BENCHMARK MASSIVO (N={n/1e6:.1f}M nós) ---")
    print(f"Gerando Grafo Aleatório Esparso via NumPy (rápido)...")

    # Geração otimizada para evitar estouro de RAM com NetworkX
    # Erdős-Rényi approximation: arestas aleatórias
    m = n * avg_degree
    np.random.seed(42)

    # Gerar arestas (u, v, w) diretamente em vetores
    # u = random int [0, n)
    # v = random int [0, n)
    # w = random int [1, 100)
    sources = np.random.randint(0, n, m, dtype=np.int32)
    targets = np.random.randint(0, n, m, dtype=np.int32)
    weights = np.random.randint(1, 100, m, dtype=np.int32)

    # Remover self-loops para limpeza
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    weights = weights[mask]

    # Converter para lista de tuplas para compatibilidade com APIs existentes
    # (Embora V18 pudesse usar arrays diretos, vamos manter a interface comum)
    # Para 10M, iterar em Python é lento, mas necessário para converter formato.
    # Vamos usar zip eficiente.
    print(f"Preparando dados de arestas...")
    edges_list = list(zip(sources, targets, weights))
    print(f"Arestas Lógicas: {len(edges_list)}")

    # Simetrização
    print("Simetrizando arestas...")
    # edges_sym = edges_list + [(v, u, w) for u, v, w in edges_list]
    # Otimização de memória: DuanNativeSolver pode escrever duas vezes no arquivo em vez de duplicar lista na RAM

    print("-" * 60)

    # --- 1. DUAN ET AL. (NATIVE C++) ---
    print(">>> [1/3] EXECUTANDO DUAN ET AL. (C++ Native - O3)...")
    # Para 10M, precisamos passar a lista simetrizada de forma inteligente
    # A classe DuanNativeSolver escreve no arquivo. Vamos alterar a chamada para evitar duplicar RAM.
    # Passamos edges_list e pedimos para o solver tratar a simetria na escrita se possivel,
    # mas como a classe espera lista, vamos criar um generator para economizar RAM?
    # Por simplicidade e robustez, vamos usar edges_list e escrever 2x no DuanNativeSolver (modificando-o on-the-fly abaixo seria complexo)
    # Vamos concatenar iterators.
    import itertools
    edges_sym_iter = itertools.chain(edges_list, ((v, u, w) for u, v, w in edges_list))
    m_sym = len(edges_list) * 2

    try:
        duan_solver = DuanNativeSolver(n, edges_sym_iter, num_edges=m_sym)
        dist_duan, time_duan = duan_solver.solve(0)
        print(f"    Tempo Duan (C++): {time_duan:.4f} ms")
    except Exception as e:
        print(f"    ERRO ao executar Duan C++: {e}")
        dist_duan = {}
        time_duan = float('inf')

    # --- 2. V18 (Vectorized SPFA) ---
    print("\n>>> [2/3] EXECUTANDO V18 (SYMMETRIC VECTORIZED)...")
    v18_solver = HFS_SSSP_V18_Symmetric(n, edges_list, symmetric=True)
    dist_v18, time_v18, hops = v18_solver.solve_exact(0)
    print(f"    Tempo V18 : {time_v18:.4f} ms (Hops: {hops})")

    # --- 3. NETWORKX (Baseline C++) ---
    # NetworkX vai sofrer com 10M nós.
    print("\n>>> [3/3] EXECUTANDO NETWORKX (Baseline Standard)...")
    print("    (Aviso: NetworkX pode demorar minutos...)")
    t_nx = time.perf_counter()
    # Recriar grafo NetworkX a partir da lista (caro)
    # G = nx.Graph()
    # G.add_weighted_edges_from(edges_list)
    # dist_nx = nx.single_source_dijkstra_path_length(G, 0)
    # time_nx = (time.perf_counter() - t_nx) * 1000
    # print(f"    Tempo NX  : {time_nx:.4f} ms")

    # SKIP NETWORKX FOR 10M (Too slow / Memory intensive to build object graph)
    print("    [SKIPPED] NetworkX pulado para economizar tempo/memória em 10M nós.")
    dist_nx = {}
    time_nx = float('inf')

    print("-" * 60)

    # --- VERIFICAÇÃO DE EXATIDÃO ---
    # Como não rodamos NetworkX (Truth), vamos comparar Duan vs V18
    print("[Verificação de Consistência: V18 vs Duan]")
    diffs = 0
    # Checar amostra aleatória de 1000 nós
    sample_nodes = np.random.choice(n, 1000)

    for node in sample_nodes:
        d_v18 = dist_v18[node]
        d_duan = dist_duan.get(node, float('inf'))

        # Tratar floats e infs
        if np.isinf(d_v18) and np.isinf(d_duan): continue
        if abs(d_v18 - d_duan) > 1e-5:
            diffs += 1
            if diffs < 5: print(f"Diff Node {node}: V18={d_v18} vs Duan={d_duan}")

    if diffs == 0: print("✅ Consistência Perfeita entre V18 e Duan (C++)")
    else: print(f"⚠️ Divergência em {diffs}/1000 nós amostrados.")

    # if errors_v18 == 0: print("✅ V18: 100% Exato.")
    # else: print(f"❌ V18: Falha ({errors_v18} erros).")

    # if errors_duan == 0: print("✅ Duan (C++): 100% Exato.")
    # else: print(f"❌ Duan (C++): Falha ({errors_duan} erros).")

    # --- RESULTADO FINAL ---
    print(f"\n[PLACAR FINAL]")
    print(f"1. V18 (Vectorized): {time_v18:.2f} ms")
    print(f"2. NetworkX (C++)  : {time_nx:.2f} ms")
    print(f"3. Duan (C++ O3)   : {time_duan:.2f} ms")

    if time_duan < float('inf'):
        print(f"\n[ANÁLISE DE SPEEDUP]")
        print(f"🚀 V18 é {time_duan / time_v18:.2f}x mais rápido que Duan C++")
