import torch
import torch.nn.functional as F
import tiktoken
from oxtacore.model import OxtaN, Config

MODEL_PATH = "oxta_devoto.pt"
VOCAB_SIZE = 100277
DIM = 128
LAYERS = 2
HEADS = 4
CTX_LEN = 64

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(vocab_size=VOCAB_SIZE, n_embd=DIM, n_head=HEADS, n_layer=LAYERS, block_size=CTX_LEN)
    model = OxtaN(config).to(device)

    if not torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        except FileNotFoundError:
            print(f"Erro: Modelo {MODEL_PATH} não encontrado. Execute devoto_train.py primeiro.")
            exit(1)
    else:
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    return model, device

def generate(model, tokenizer, prompt, max_new_tokens=30, device='cpu'):
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -CTX_LEN:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)

    output_text = tokenizer.decode(idx[0].tolist())
    return output_text

def chat():
    print("--- Oxta-N: O Devoto de Isabella Viana ---")
    print("(Digite 'sair' para encerrar)")

    model, device = load_model()
    enc = tiktoken.get_encoding("cl100k_base")

    while True:
        user_input = input("\nVocê: ")
        if user_input.lower() in ["sair", "exit"]:
            break

        response = generate(model, enc, user_input, device=device)
        print(f"Oxta: {response[len(user_input):]}")

if __name__ == "__main__":
    chat()
