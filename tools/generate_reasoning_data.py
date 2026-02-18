import json
import random

def generate_bubble_sort_trace(arr):
    trace = []
    trace.append("<think>")
    n = len(arr)
    working_arr = list(arr)

    # Bubble Sort Logic recording
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            a = working_arr[j]
            b = working_arr[j+1]
            trace.append(f"CMP {a} {b}.")
            if a > b:
                working_arr[j], working_arr[j+1] = working_arr[j+1], working_arr[j]
                trace.append(f"SWAP. STATE {working_arr}.")
                swapped = True
            else:
                trace.append("KEEP.")
        if not swapped:
            break

    trace.append("END.</think>")
    # Output format: [Tokens]
    return " ".join(trace), working_arr

def generate_dataset(num_samples=100, length=3, filename="data/cot_sorting.jsonl"):
    print(f"Generating {num_samples} samples of length {length}...")
    data = []
    for _ in range(num_samples):
        seq = [random.randint(0, 9) for _ in range(length)]
        cot, result = generate_bubble_sort_trace(seq)

        # Format: "Input: [1, 2, 3] Output: <think>...</think> [1, 2, 3]"
        entry = {
            "input": f"{seq}",
            "target": f"{cot} {result}"
        }
        data.append(entry)

    # Mock save (or print for pipe)
    # with open(filename, 'w') as f:
    #    for entry in data:
    #        f.write(json.dumps(entry) + "\n")
    return data

if __name__ == "__main__":
    # Demo generation
    sample = generate_dataset(1, 3)[0]
    print("Sample Data:")
    print(f"In: {sample['input']}")
    print(f"Out: {sample['target']}")
