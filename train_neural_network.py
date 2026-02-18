#!/usr/bin/env python3
"""
NSOS Training Script - Demonstração de Treinamento de Rede Neural
Treina um modelo simples em tarefas de aprendizado para verificar que o gradiente 
flui corretamente e a loss diminui ao longo das epochs.
"""

import numpy as np
import time
import sys

# Configuração
np.random.seed(42)

print("=" * 60)
print("🧠 NSOS Neural Network Training Demo")
print("=" * 60)

# =============================================================================
# 1. Implementação do Modelo (BitLinear MLP)
# =============================================================================

def quantize_ternary(w):
    """Quantiza pesos para {-1, 0, +1} (1.58-bit)"""
    scale = np.abs(w).mean() + 1e-8
    w_scaled = w / scale
    w_ternary = np.round(np.clip(w_scaled, -1, 1))
    return w_ternary, scale

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (e_x.sum(axis=-1, keepdims=True) + 1e-8)

def cross_entropy_loss(pred, target):
    """Cross-entropy loss com one-hot targets"""
    eps = 1e-8
    return -np.sum(target * np.log(pred + eps)) / pred.shape[0]

class BitLinearLayer:
    """Camada Linear com quantização 1.58-bit nos pesos"""
    
    def __init__(self, in_features, out_features, name=""):
        self.name = name
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features, dtype=np.float32)
        
        # Gradientes
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache para backward
        self.input_cache = None
        self.W_ternary = None
        self.scale = None
    
    def forward(self, x):
        self.input_cache = x
        # Quantiza pesos para forward
        self.W_ternary, self.scale = quantize_ternary(self.W)
        # Forward com pesos quantizados
        out = x @ self.W_ternary + self.b
        return out
    
    def backward(self, grad_out):
        """Backward pass - usa STE (Straight-Through Estimator) para gradientes através da quantização"""
        # Gradientes dos parâmetros
        self.dW = self.input_cache.T @ grad_out
        self.db = grad_out.sum(axis=0)
        
        # Gradiente para camada anterior (usa W original para gradiente, não quantizado)
        grad_in = grad_out @ self.W.T
        return grad_in

class BitLinearMLP:
    """MLP com camadas BitLinear"""
    
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(BitLinearLayer(layer_sizes[i], layer_sizes[i+1], f"layer_{i}"))
        
        print(f"📦 Modelo criado: {layer_sizes}")
        total_params = sum(l.W.size + l.b.size for l in self.layers)
        print(f"   Total de parâmetros: {total_params:,}")
    
    def forward(self, x):
        self.activations = [x]
        
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:  # ReLU em todas menos última
                x = relu(x)
            self.activations.append(x)
        
        return softmax(x)
    
    def backward(self, pred, target):
        """Backward pass completo"""
        # Gradiente inicial: dL/d(logits) para softmax + cross-entropy
        grad = pred - target  # Simplificação para softmax + CE
        grad = grad / pred.shape[0]  # Normaliza por batch
        
        # Backward através das camadas (reverso)
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:  # ReLU gradient (exceto última)
                pre_activation = self.activations[i+1]
                grad = grad * relu_grad(self.layers[i].forward(self.activations[i]))
            
            grad = self.layers[i].backward(grad)
    
    def update(self, learning_rate):
        """Atualiza pesos com SGD"""
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db
            # Zero gradients
            layer.dW.fill(0)
            layer.db.fill(0)

# =============================================================================
# 2. Tarefa de Treinamento: Classificação de Padrões
# =============================================================================

def create_dataset(n_samples=1000, noise=0.1):
    """Cria dataset de classificação de 4 padrões (como XOR expandido)"""
    X = np.random.randn(n_samples, 4).astype(np.float32)
    
    # 4 classes baseadas em combinações de features
    # Classe 0: x0 > 0 AND x1 > 0
    # Classe 1: x0 < 0 AND x1 > 0
    # Classe 2: x0 > 0 AND x1 < 0
    # Classe 3: x0 < 0 AND x1 < 0
    
    y = np.zeros((n_samples, 4), dtype=np.float32)
    for i in range(n_samples):
        if X[i, 0] > 0 and X[i, 1] > 0:
            y[i, 0] = 1
        elif X[i, 0] < 0 and X[i, 1] > 0:
            y[i, 1] = 1
        elif X[i, 0] > 0 and X[i, 1] < 0:
            y[i, 2] = 1
        else:
            y[i, 3] = 1
    
    # Adiciona ruído
    X += np.random.randn(*X.shape).astype(np.float32) * noise
    
    return X, y

def accuracy(pred, target):
    """Calcula acurácia"""
    pred_class = np.argmax(pred, axis=1)
    target_class = np.argmax(target, axis=1)
    return (pred_class == target_class).mean() * 100

# =============================================================================
# 3. Loop de Treinamento
# =============================================================================

def train():
    print("\n" + "=" * 60)
    print("📊 Preparando Dataset...")
    print("=" * 60)
    
    # Dataset
    X_train, y_train = create_dataset(n_samples=2000, noise=0.2)
    X_test, y_test = create_dataset(n_samples=500, noise=0.2)
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Classes: {y_train.shape[1]}")
    
    # Modelo
    print("\n" + "=" * 60)
    print("🏗️ Construindo Modelo BitLinear...")
    print("=" * 60)
    
    model = BitLinearMLP([4, 32, 16, 4])  # Input -> Hidden1 -> Hidden2 -> Output
    
    # Hiperparâmetros
    epochs = 100
    batch_size = 32
    learning_rate = 0.05
    lr_decay = 0.99  # Decay a cada epoch
    
    print(f"\n   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    
    # Treinamento
    print("\n" + "=" * 60)
    print("🚀 Iniciando Treinamento...")
    print("=" * 60)
    print(f"\n{'Epoch':>6} | {'Loss':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Time':>8}")
    print("-" * 60)
    
    n_batches = len(X_train) // batch_size
    history = {'loss': [], 'train_acc': [], 'test_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        
        # Shuffle
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward
            pred = model.forward(X_batch)
            loss = cross_entropy_loss(pred, y_batch)
            epoch_loss += loss
            
            # Backward
            model.backward(pred, y_batch)
            
            # Update
            model.update(learning_rate)
        
        # Métricas
        avg_loss = epoch_loss / n_batches
        
        train_pred = model.forward(X_train)
        train_acc = accuracy(train_pred, y_train)
        
        test_pred = model.forward(X_test)
        test_acc = accuracy(test_pred, y_test)
        
        epoch_time = time.time() - epoch_start
        
        history['loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"{epoch+1:>6} | {avg_loss:>10.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}% | {epoch_time:>7.3f}s")
    
    total_time = time.time() - start_time
    
    # Resultados finais
    print("\n" + "=" * 60)
    print("📈 Resultados do Treinamento")
    print("=" * 60)
    
    print(f"\n   ⏱️  Tempo total: {total_time:.2f}s")
    print(f"   📉 Loss inicial: {history['loss'][0]:.4f}")
    print(f"   📉 Loss final: {history['loss'][-1]:.4f}")
    print(f"   📊 Redução de loss: {((history['loss'][0] - history['loss'][-1]) / history['loss'][0] * 100):.1f}%")
    print(f"\n   🎯 Acurácia Train: {history['train_acc'][-1]:.2f}%")
    print(f"   🎯 Acurácia Test: {history['test_acc'][-1]:.2f}%")
    
    # Verificação de aprendizado
    print("\n" + "=" * 60)
    print("✅ Verificação de Aprendizado")
    print("=" * 60)
    
    learned = history['loss'][-1] < history['loss'][0] * 0.5
    good_accuracy = history['test_acc'][-1] > 70
    no_overfit = abs(history['train_acc'][-1] - history['test_acc'][-1]) < 20
    
    print(f"\n   [{'✓' if learned else '✗'}] Loss diminuiu significativamente")
    print(f"   [{'✓' if good_accuracy else '✗'}] Acurácia > 70%")
    print(f"   [{'✓' if no_overfit else '✗'}] Sem overfitting severo")
    
    if learned and good_accuracy:
        print("\n   🎉 MODELO APRENDEU COM SUCESSO!")
    else:
        print("\n   ⚠️ Modelo pode precisar de mais treinamento ou ajustes")
    
    # Demonstração de inferência
    print("\n" + "=" * 60)
    print("🔮 Demonstração de Inferência")
    print("=" * 60)
    
    test_samples = [
        [1.0, 1.0, 0.5, 0.5],   # Esperado: Classe 0
        [-1.0, 1.0, 0.5, 0.5],  # Esperado: Classe 1
        [1.0, -1.0, 0.5, 0.5],  # Esperado: Classe 2
        [-1.0, -1.0, 0.5, 0.5], # Esperado: Classe 3
    ]
    
    expected = [0, 1, 2, 3]
    
    for i, sample in enumerate(test_samples):
        x = np.array([sample], dtype=np.float32)
        pred = model.forward(x)
        pred_class = np.argmax(pred)
        confidence = pred[0, pred_class] * 100
        correct = "✓" if pred_class == expected[i] else "✗"
        print(f"   Input: {sample[:2]} -> Classe {pred_class} ({confidence:.1f}% confiança) {correct}")
    
    print("\n" + "=" * 60)
    print("✅ Treinamento Completo!")
    print("=" * 60)
    
    return history

if __name__ == "__main__":
    history = train()
