"""Small, bounded Portuguese tasks. Generator and verifier use separate algorithms.

The verifier parses the delivered prompt/answer, never a generator-provided answer
key. This verifies only this grammar, not general mathematical reasoning.
"""
from collections import Counter
from decimal import Decimal
import hashlib
import json
import random
import re

VERSION = "ptbr-exact-tasks-v1"
FAMILIES = ("arithmetic", "inventory", "units", "sorting", "extraction")


def verify(prompt, answer):
    """Return a canonical problem identity or reject malformed/incorrect output."""
    if not isinstance(prompt, str) or not isinstance(answer, str):
        raise ValueError("Expected text")
    if len(prompt) > 2000 or len(answer) > 1000:
        raise ValueError("Oversized task")
    match = re.fullmatch(r"Calcule (\d{1,4}) ([+*\-]) (\d{1,4})\. Responda somente com o inteiro\.", prompt)
    if match:
        left, op, right = match.groups()
        a, b = Decimal(left), Decimal(right)
        expected = {"+": lambda: a+b, "-": lambda: a-b, "*": lambda: a*b}[op]()
        if not re.fullmatch(r"-?\d{1,9}", answer) or Decimal(answer) != expected:
            raise ValueError("Incorrect arithmetic")
        operands = [int(a), int(b)]
        if op in ("+", "*"):
            operands.sort()  # a+b and b+a cannot cross the split boundary.
        identity = ["arithmetic", op, operands]
    elif (match := re.fullmatch(
            r"Um estoque tinha (\d{1,4}) caixas\. Recebeu (\d{1,4}) caixas e despachou (\d{1,4}) caixas\. "
            r"Quantas caixas restaram\? Responda somente com o inteiro\.", prompt)):
        before, incoming, outgoing = map(Decimal, match.groups())
        if outgoing > before+incoming or not re.fullmatch(r"\d{1,5}", answer):
            raise ValueError("Invalid inventory")
        if Decimal(answer)+outgoing != before+incoming:
            raise ValueError("Inventory conservation failed")
        identity = ["inventory", *map(int, (before, incoming, outgoing))]
    elif (match := re.fullmatch(
            r"Converta (\d{1,4}) (metros|quilogramas|minutos) para (centímetros|gramas|segundos)\. "
            r"Responda somente com o inteiro\.", prompt)):
        value, source, target = match.groups()
        factors = {("metros", "centímetros"): Decimal(100),
                   ("quilogramas", "gramas"): Decimal(1000),
                   ("minutos", "segundos"): Decimal(60)}
        factor = factors.get((source, target))
        if factor is None or not re.fullmatch(r"\d{1,8}", answer) or Decimal(answer)/factor != Decimal(value):
            raise ValueError("Incorrect unit conversion")
        identity = ["units", int(value), source, target]
    elif prompt.startswith("Ordene em ordem crescente a lista JSON "):
        match = re.fullmatch(r"Ordene em ordem crescente a lista JSON (\[[\d, \-]+\])\. Responda somente com a lista JSON\.", prompt)
        if not match:
            raise ValueError("Invalid sorting prompt")
        numbers, result = json.loads(match[1]), json.loads(answer)
        if (not isinstance(result, list) or not 3 <= len(numbers) <= 10 or
                any(type(n) is not int or abs(n) > 9999 for n in numbers+result) or
                Counter(numbers) != Counter(result) or
                any(a > b for a, b in zip(result, result[1:]))):
            raise ValueError("Incorrect ordered multiset")
        # All permutations of a multiset are one problem for holdout isolation.
        identity = ["sorting", sorted(numbers)]
    elif (match := re.fullmatch(
            r'No objeto JSON (\{[^\n]+\}), qual é o valor da chave "(azul|verde|amarelo)"\? '
            r'Responda somente com o inteiro\.', prompt)):
        pairs = json.loads(match[1], object_pairs_hook=list)
        if (not isinstance(pairs, list) or len(pairs) != 3 or
                {p[0] for p in pairs} != {"azul", "verde", "amarelo"} or
                any(type(p[1]) is not int or not 0 <= p[1] <= 9999 for p in pairs)):
            raise ValueError("Invalid object")
        obj = dict(pairs)
        if not re.fullmatch(r"\d{1,4}", answer) or int(answer) != obj[match[2]]:
            raise ValueError("Incorrect extraction")
        identity = ["extraction", sorted(obj.items()), match[2]]
    else:
        raise ValueError("Unsupported task grammar")
    key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"version": VERSION, "family": identity[0], "problem_id": key,
            "split": "eval" if int(key[:8], 16) % 100 < 5 else "train"}


def generate(seed, index):
    """Deterministic by index; no shared RNG state or external model is needed."""
    rng = random.Random(f"{VERSION}:{seed}:{index}")
    family = FAMILIES[index % len(FAMILIES)]
    a, b = rng.randrange(1, 1000), rng.randrange(1, 1000)
    if family == "arithmetic":
        op = rng.choice(("+", "-", "*"))
        value = a+b if op == "+" else a-b if op == "-" else a*b
        prompt = f"Calcule {a} {op} {b}. Responda somente com o inteiro."
        answer = str(value)
    elif family == "inventory":
        sent = rng.randrange(a+b+1)
        prompt = (f"Um estoque tinha {a} caixas. Recebeu {b} caixas e despachou {sent} caixas. "
                  "Quantas caixas restaram? Responda somente com o inteiro.")
        answer = str(a+b-sent)
    elif family == "units":
        source, target, factor = rng.choice((("metros", "centímetros", 100),
                                           ("quilogramas", "gramas", 1000),
                                           ("minutos", "segundos", 60)))
        a = rng.randrange(1, 10000)
        prompt = f"Converta {a} {source} para {target}. Responda somente com o inteiro."
        answer = str(a*factor)
    elif family == "sorting":
        numbers = [rng.randrange(-999, 1000) for _ in range(rng.randrange(3, 9))]
        prompt = f"Ordene em ordem crescente a lista JSON {json.dumps(numbers)}. Responda somente com a lista JSON."
        answer = json.dumps(sorted(numbers))
    else:
        obj = {key: rng.randrange(10000) for key in ("azul", "verde", "amarelo")}
        key = rng.choice(list(obj))
        prompt = f'No objeto JSON {json.dumps(obj)}, qual é o valor da chave "{key}"? Responda somente com o inteiro.'
        answer = str(obj[key])
    return {"prompt": prompt, "answer": answer}


def render(record):
    return "Pergunta: " + record["prompt"] + "\nResposta: " + record["answer"]
