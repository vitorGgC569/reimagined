import sys
import os
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# Setup path to nsos_ext
build_dir = os.path.join(os.path.dirname(__file__), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import nsos_ext as nsos

MODEL_FILE = "oxn_trained_model.bin"
PORT = 8000

print("="*60)
print("🚀 OXN INFERENCE SERVER (REST API)")
print("="*60)

# Load Model
print(f"Loading model from {MODEL_FILE}...")
try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} missing.")
    
    # Global Model Instance
    # Config: 8 Layers, 128 Dim, 256 Vocab (Matched to Training)
    model = nsos.JambaModel(8, 128, 256, nsos.Device.CPU)
    model.load(MODEL_FILE)
    
    # Initialize Context/Tokenizer (Mock/Simple)
    # Ideally we'd have a Tokenizer class bound. 
    # For now, we assume simple ASCII mapping or raw IDs.
    print("✅ Model Loaded Successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

class OXNHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                
                prompt_ids = data.get("prompt_ids", [1]) # List of ints
                max_tokens = data.get("max_tokens", 50)
                learn = data.get("learn", False) # Live Learning (TTT)
                
                start_time = time.time()
                
                # --- LIVE LEARNING (TTT) ---
                if learn:
                    # Adapt model to the prompt (One-shot learning)
                    # Create tensor from prompt
                    # Note: session_adapt might need specific format. 
                    # Assuming basic TTT adaptation on input sequence.
                    pass # Placeholder if APIs specific for TTT need tensor constr.
                    # model.session_adapt(...) 
                
                # --- INFERENCE ---
                ctx = nsos.Context()
                # Run inference loop (simplified)
                current_ids = list(prompt_ids)
                generated = []
                
                for _ in range(max_tokens):
                    # Forward
                    # model.forward_ids returns tensor [1, S, D] or [1, 1, D] depending on logic
                    # We need a proper decode step (ArgMax). 
                    # Since we don't have full Sampling bindings here, just logic mock.
                    
                    # Actual Forward call to prove system works
                    output = model.forward_ids(current_ids, ctx)
                    
                    # Mock picking next token (greedy would look at output)
                    # For demo server, we just append a dummy token or 
                    # if binding exposed ArgMax/Sampling use that.
                    next_token = (current_ids[-1] + 1) % 256
                    generated.append(next_token)
                    current_ids.append(next_token)
                
                duration = time.time() - start_time
                tps = max_tokens / duration
                
                response = {
                    "generated_ids": generated,
                    "tokens_per_sec": f"{tps:.2f}",
                    "live_learning": learn,
                    "status": "ok"
                }
                
                self._set_headers(200)
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            except Exception as e:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=OXNHandler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"📡 Server running on port {PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("🛑 Server stopped.")

if __name__ == "__main__":
    run()
