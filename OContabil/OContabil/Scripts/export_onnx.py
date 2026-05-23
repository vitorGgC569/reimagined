import sys
import json
import codecs
from pathlib import Path

# Force UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Tentamos importar. Se nao conseguir, avisamos o usuario.
try:
    import torch
    from gliner2 import GLiNER2
except ImportError:
    print("ERRO: Pacotes faltando. Execute: python -m pip install gliner2 onnxruntime onnx torch torchvision torchaudio")
    sys.exit(1)

def export_to_onnx(model_name="fastino/gliner2-base-v1", output_dir="Models"):
    print(f"Carregando modelo {model_name}...")
    model = GLiNER2.from_pretrained(model_name)
    
    # Salvar configuracao original para C# poder ler os labels e config
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    config_path = out_path / "gliner_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model.config.to_dict(), f, indent=2)
        
    onnx_path = out_path / "gliner2.onnx"
    
    print(f"Exportando para {onnx_path} (Isso pode demorar alguns minutos)...")
    
    # Export explicitly to ONNX
    try:
        model.export_model_to_onnx(str(onnx_path))
        print("Exportacao CONCLUIDA! O modelo ONNX esta pronto para ser usado no C#.")
    except AttributeError:
        print("A versao instalada do gliner nao possui o metodo export_model_to_onnx ou ele falhou.")
        print("Verifique a documentacao oficial do GLiNER 2 para os parametros de conversao ONNX.")
        sys.exit(1)

if __name__ == "__main__":
    export_to_onnx()
