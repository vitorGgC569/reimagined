import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
import nsos_ext
import numpy as np

print("Verifying NSOS Model...")
model = nsos_ext.JambaModel(4, 64)
head = nsos_ext.BitFastKANLayer(64, 128)
tokenizer = nsos_ext.Tokenizer()

text = "System Check"
tokens = tokenizer.encode(text)
input_tensor = nsos_ext.Tensor.random([len(tokens), 64], nsos_ext.Device.CPU)

latent = model.forward_embedding(input_tensor)
logits = head.forward(latent)
logits_np = np.array(logits, copy=False)
pred_ids = np.argmax(logits_np, axis=1).tolist()
output = tokenizer.decode(pred_ids)

print(f"Input: {text}")
print(f"Model Output (Untrained): {output}")
print("Verification Successful.")
