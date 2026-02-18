import os
import random

def generate_circuit():
    types = ["DIVIDER", "FILTER_RC", "LED_DRV", "LOGIC_AND", "LOGIC_OR"]
    ctype = random.choice(types)
    
    if ctype == "DIVIDER":
        r1, r2 = random.randint(1, 100), random.randint(1, 100)
        net = f"V1 1 0 5V;R1 1 2 {r1}k;R2 2 0 {r2}k"
        pos = f"V1(0,10);R1(10,10);R2(10,0);W(0,10,10,10);W(10,10,10,0)"
        return f"REQ:voltage_divider_{r1}_{r2}|NET:{net}|POS:{pos}\n"
    
    elif ctype == "FILTER_RC":
        r, c = random.randint(1, 10), random.randint(1, 100)
        net = f"IN 1 0;R1 1 2 {r}k;C1 2 0 {c}n;OUT 2 0"
        pos = f"IN(0,5);R1(10,5);C1(10,0);OUT(20,5);W(0,5,10,5);W(10,5,10,0);W(10,5,20,5)"
        return f"REQ:rc_filter_{r}k_{c}n|NET:{net}|POS:{pos}\n"
    
    elif ctype == "LED_DRV":
        v = random.choice([5, 9, 12])
        r = random.randint(220, 1000)
        net = f"V1 1 0 {v}V;R1 1 2 {r};D1 2 0 LED"
        pos = f"V1(0,10);R1(10,10);D1(10,0);W(0,10,10,10);W(10,10,10,0)"
        return f"REQ:led_driver_{v}v|NET:{net}|POS:{pos}\n"
    
    else: # LOGIC GATES
        net = "IN1 1 0;IN2 2 0;U1 1 2 3 GATE;OUT 3 0"
        pos = "IN1(0,10);IN2(0,0);U1(10,5);OUT(20,5);W(0,10,10,10);W(0,0,10,0);W(10,5,20,5)"
        return f"REQ:logic_block_{ctype}|NET:{net}|POS:{pos}\n"

def main():
    os.makedirs("circuit_data", exist_ok=True)
    with open("circuit_data/train.raw", "w") as f:
        print("🛠️ Generating 50,000 SOTA circuit examples...")
        for _ in range(50000):
            f.write(generate_circuit())
    print("✅ Dataset generated: circuit_data/train.raw")

if __name__ == "__main__":
    main()
