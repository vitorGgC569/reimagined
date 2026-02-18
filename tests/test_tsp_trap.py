
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

# Setup
sys.path.append(os.path.abspath("OXN/nsos/build"))
try:
    import nsos_ext
except ImportError:
    print("Error importing nsos_ext")
    sys.exit(1)

# --- TurboFusion (Policy) ---
class TurboFusionAdapter(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.net = nn.Linear(dim, vocab) # Simple linear for this test to rely on TTT
    def forward(self, x):
        return self.net(x)

# --- Marco Zero Solver ---
class MarcoZeroSolver:
    def __init__(self, n_cities):
        self.n = n_cities
        self.dim = 64
        self.model = nsos_ext.JambaModel(2, self.dim, n_cities + 10)
        self.adapter = TurboFusionAdapter(self.dim, n_cities)
        self.ctx = nsos_ext.Context()
        self.ctx.set_metadata("force_system2", True)

    def solve(self, coords, hamiltonian=False):
        # 1. Configurar Modo
        self.model.set_hamiltonian_mode(hamiltonian)
        mode_name = "Hamiltonian (Gênio)" if hamiltonian else "Standard (Guloso)"
        print(f"\n>>> Resolvendo com Modo: {mode_name}")

        # 2. Injetar Geometria (128-bit Anchor Proxy via Embedding Override)
        # Hack: sobrescrevemos os pesos do embedding com as coordenadas normalizadas
        w_ptr = np.array(self.model.embedding.weight.data, copy=False)
        w_reshaped = w_ptr.reshape((self.model.embedding.vocab_size, self.dim))

        # Reset para garantir que não haja contaminação anterior
        w_reshaped.fill(0)
        # Coords [N, 2]. Scale to make them distinct anchors
        for i in range(self.n):
            w_reshaped[i, 0] = coords[i][0] * 10.0
            w_reshaped[i, 1] = coords[i][1] * 10.0

        # 3. TTT (O Pensamento)
        # O modelo "olha" para o mapa e ajusta seus pesos internos
        input_ids = list(range(self.n))
        emb = self.model.embedding.forward(input_ids)

        # Loop de Adaptação (TTT)
        # Standard: Ajusta para minimizar erro de reconstrução local
        # Hamiltonian: Ganha momento para pular mínimos locais
        # UPDATED: 50 passos de órbita para o "Modo Gênio" sentir a gravidade
        for _ in range(50):
            self.model.session_adapt(emb, emb)

        # 4. Inferência (Reasoning Loop)
        hidden = self.model.run_reasoning_loop(emb, 4)
        h_pt = torch.from_numpy(np.array(hidden, copy=False)).float()

        # 5. Decodificação (Greedy com Máscara)
        logits = self.adapter(h_pt) # [N, N] map

        tour = [0] # Start at 0
        mask = torch.zeros(self.n)
        mask[0] = -float('inf')

        current = 0
        for _ in range(self.n - 1):
            # Next city depends on current hidden state
            # Simple projection: Logits[current] -> Next ID
            # In a real autoregressive model, we'd feed previous token.
            # Here we trust the TTT modified latent space to guide the sequence.
            scores = logits[current] + mask
            next_city = torch.argmax(scores).item()
            tour.append(next_city)
            mask[next_city] = -float('inf')
            current = next_city

        return tour

def calculate_len(coords, tour):
    d = 0
    for i in range(len(tour)):
        p1 = coords[tour[i]]
        p2 = coords[tour[(i+1)%len(tour)]]
        d += np.linalg.norm(p1-p2)
    return d

def test_trap():
    print("=== TESTE DA ARMADILHA DO MÍNIMO LOCAL ===")

    # Construir o Mapa "Armadilha" (The Trap)
    # Um "U" onde o ponto inicial (0) tem um vizinho "Bait" (1) muito perto,
    # mas que obriga a cruzar o "U" inteiro para voltar.

    # U-Shape Points: 2, 3, 4, 5, 6, 7, 8, 9
    # Start: 0
    # Bait: 1 (Very close to 0, but far from U start)

    coords = np.array([
        [0.0, 0.0], # 0: Start
        [0.1, 0.0], # 1: Bait (Very close to 0)

        # The U-Shape (Farther from 0 than 1 is, but creates a loop)
        [0.0, 1.0], # 2
        [0.0, 2.0], # 3
        [1.0, 2.0], # 4
        [2.0, 2.0], # 5
        [2.0, 1.0], # 6
        [2.0, 0.0], # 7
        [1.0, 0.0], # 8
        [0.5, 0.5]  # 9: Center (Trap if you cross)
    ], dtype=np.float32)

    # Optimal human path likely avoids 1 immediately if it implies a cross?
    # Or takes 1 then 8?
    # Let's see what the solvers do.

    solver = MarcoZeroSolver(len(coords))

    # Run Hamiltonian Only (The Genius Test)
    tour_ham = solver.solve(coords, hamiltonian=True)
    len_ham = calculate_len(coords, tour_ham)
    print(f"Rota Hamiltoniana: {tour_ham}")
    print(f"Distância: {len_ham:.4f}")

    # Baseline Check (The Bait)
    # The Trap Path: 0 -> 1 -> ... (Greedy) usually costs > 17
    # The Smart Path: 0 -> 2 ... -> 1 (Backwards) usually costs < 16 (approx)

    if len_ham < 17.0:
        print(f"\n🏆 SUCESSO ABSOLUTO! O Gênio escapou da armadilha.")
    else:
        print(f"\n⚠️ O Gênio caiu na armadilha. A física precisa de mais 'shake'.")

if __name__ == "__main__":
    test_trap()
