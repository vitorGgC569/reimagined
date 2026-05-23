# export_simple.py
import torch
from gliner2 import GLiNER2
from transformers import AutoTokenizer
import json
import os
import sys
from pathlib import Path

# Force UTF-8 for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def export_simple(model_name="fastino/gliner2-base-v1", output_dir="./Models"):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print(f"Exportando {model_name} de forma simplificada...")
    model = GLiNER2.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Salvar config
    with open(out_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    
    tokenizer.save_pretrained(str(out_path))
    
    # Usar Tensores básicos para evitar FakeTensor/AttributeError em versões novas do torch
    dummy_text = "Nota Fiscal exemplo"
    tokens = tokenizer(dummy_text, return_tensors="pt")
    
    onnx_file = out_path / "model.onnx"
    
    print(f"Exportando para {onnx_file}...")
    
    # Desativar JIT para exportação mais simples
    with torch.no_grad():
        torch.onnx.export(
            model,
            (tokens["input_ids"], tokens["attention_mask"]),
            str(onnx_file),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"}
            },
            opset_version=13, # Tentando opset mais estável
            do_constant_folding=True
        )
    print("CONCLUÍDO!")

if __name__ == "__main__":
    export_simple()
