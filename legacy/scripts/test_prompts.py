
import sys
import os
import numpy as np

# Setup NSOS path
build_dir = os.path.join(os.getcwd(), "OXN", "build", "Release")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)
import nsos_ext as nsos

def test_p(prompt, model, head, vocab_size=256):
    ids = [min(ord(c), vocab_size-1) for c in prompt]
    ctx = nsos.Context()
    h = model.forward_ids(ids, ctx)
    # h is [1, len(ids), dim]
    hidden_2d = h.reshape([len(ids), 128])
    logits = head.forward(hidden_2d)
    logits_np = logits.cpu().numpy()
    next_id = int(np.argmax(logits_np[-1]))
    return next_id

def main():
    # Attempt to load with vocab 256 (WikiText)
    try:
        model = nsos.JambaModel(4, 128, 256)
        model.load("oxn_main_trunk.bin")
        print("✅ Trunk loaded (Vocab 256)")
        head_file = "head_fixed.npy"
        head = nsos.BitFastKANLayer(128, 256)
        if os.path.exists(head_file):
            w = np.load(head_file, allow_pickle=True).item()
            head.base_weight.data.copy_from(nsos.Tensor.from_blob(w['base'].ctypes.data, [256, 128], nsos.Device.CPU))
            print("✅ Head loaded.")
        
        prompts = ["Napoleon ", "In the ", "Once "]
        for p in prompts:
            nid = test_p(p, model, head, 256)
            print(f"Prompt: '{p}' -> Next Token ID: {nid} ('{chr(nid) if 32 <= nid <= 126 else f'[{nid}]'}')")
            
    except Exception as e:
        print(f"⚠️ Failed with Vocab 256: {e}. Trying Vocab 128 (Chip Router)...")
        try:
            model = nsos.JambaModel(3, 128, 128)
            model.load("router_trunk.bin")
            print("✅ Trunk loaded (Vocab 128)")
            # Router head has 4 actions
            head = nsos.BitFastKANLayer(128, 4)
            head_base = np.load("head_base.npy")
            head.base_weight.data.copy_from(nsos.Tensor.from_blob(head_base.ctypes.data, [4, 128], nsos.Device.CPU))
            print("✅ Head loaded.")
            
            prompts = ["0,0,1,0,32,32 ", "16,16,0,0 "]
            for p in prompts:
                # For router, we just pass IDs even if it doesn't make sense as text
                nid = test_p(p, model, head, 128)
                print(f"Prompt: '{p}' -> Next Action: {nid}")
        except Exception as e2:
             print(f"❌ Failed all loads: {e2}")

if __name__ == "__main__":
    main()
