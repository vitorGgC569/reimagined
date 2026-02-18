
import random
import json
import os

def generate_scientific_curriculum(output_file="data/science_curriculum.jsonl", num_samples=1000):
    print(f"Generating Scientific Curriculum: {num_samples} samples...")

    os.makedirs("data", exist_ok=True)

    data = []

    # 1. Math Problems (Step-by-Step)
    for _ in range(num_samples // 3):
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        op = random.choice(['+', '-', '*'])

        if op == '+': res = a + b
        elif op == '-': res = a - b
        else: res = a * b

        prompt = f"Calculate {a} {op} {b}."
        cot = f"<think> {a} {op} {b} is {res}. </think>"
        target = f"{res}"

        data.append({"type": "math", "input": prompt, "target": cot + target})

    # 2. Chemistry (SMILES)
    atoms = ['C', 'O', 'N']
    for _ in range(num_samples // 3):
        mol_len = random.randint(3, 10)
        mol = "".join(random.choices(atoms, k=mol_len))
        # Valid SMILES structure is complex, using pseudo-SMILES for topology test
        # Real curriculum uses RDKit
        prompt = f"Analyze molecule: {mol}"
        cot = f"<think> This molecule has {mol.count('C')} Carbons. </think>"
        target = "Organic Compound."
        data.append({"type": "chem", "input": prompt, "target": cot + target})

    # 3. Physics (Trajectories)
    for _ in range(num_samples // 3):
        t = random.randint(1, 10)
        # s = 0.5 * g * t^2
        dist = 0.5 * 9.8 * (t**2)
        prompt = f"Free fall distance after {t} seconds?"
        cot = f"<think> d = 0.5 * g * t^2. d = 0.5 * 9.8 * {t*t} = {dist:.2f}. </think>"
        target = f"{dist:.2f}m"
        data.append({"type": "phys", "input": prompt, "target": cot + target})

    with open(output_file, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

    print("Done.")

if __name__ == "__main__":
    generate_scientific_curriculum()
