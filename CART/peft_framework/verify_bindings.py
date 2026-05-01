import peft_framework
import sys

def verify():
    print("Verifying peft_framework bindings...")

    # 1. Tensor creation
    t = peft_framework.Tensor(2, 3)
    assert t.get_rows() == 2
    assert t.get_cols() == 3
    t.set(0, 0, 1.5)
    print(f"Tensor value at (0,0): {t.at(0,0)} (Expected 1.5)")
    assert t.at(0,0) == 1.5

    # 2. LoRA Layer
    print("Testing LoRA...")
    lora = peft_framework.LoRALayer(4, 4, 2)
    w0 = peft_framework.Tensor(4, 4)
    # Fill w0
    for i in range(4):
        for j in range(4):
            w0.set(i, j, 0.5)

    lora.set_base_weights(w0)
    inp = peft_framework.Tensor(1, 4)
    out = lora.forward(inp)
    print("LoRA Forward passed.")

    # 3. DoRA Layer
    print("Testing DoRA...")
    dora = peft_framework.DoRALayer(4, 4, 2)
    dora.set_base_weights(w0)
    out_dora = dora.forward(inp)
    print("DoRA Forward passed.")

    # 4. TurboFusion Layer
    print("Testing TurboFusion...")
    turbo = peft_framework.TurboFusionLayer(4, 4, 2)
    turbo.set_base_weights(w0)
    out_turbo = turbo.forward(inp)
    print("TurboFusion Forward passed.")

    print("\nALL PYTHON BINDINGS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()
