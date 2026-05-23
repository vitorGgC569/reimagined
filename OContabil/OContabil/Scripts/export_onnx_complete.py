# export_onnx_complete.py
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

def export_gliner_onnx(model_name="fastino/gliner2-base-v1", output_dir="./Models"):
    """Exporta modelo GLiNER2 para ONNX com tokenizer."""
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print(f"Carregando modelo e tokenizer para {model_name}...")
    # Carregar modelo e tokenizer
    model = GLiNER2.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Salvar config
    config = model.config.to_dict() if hasattr(model, 'config') else {}
    config_file = out_path / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    # Exportar tokenizer
    print(f"Salvando tokenizer em {output_dir}...")
    tokenizer.save_pretrained(str(out_path))
    
    # Criar dummy input para exportação
    dummy_text = "Nota Fiscal emitida por Empresa Exemplo LTDA CNPJ 12.345.678/0001-90"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", 
                       truncation=True, max_length=512)
    
    # Exportar para ONNX
    onnx_file = out_path / "model.onnx"
    print(f"Exportando modelo para {onnx_file} (isso pode levar alguns minutos)...")
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_file),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print(f"Exportação CONCLUÍDA!")
    print(f"Arquivos gerados em {output_dir}: model.onnx, config.json, tokenizer.json")

if __name__ == "__main__":
    export_gliner_onnx()
