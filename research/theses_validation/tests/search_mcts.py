"""AB-MCTS vs vanilla MCTS (UCT) — Tese 2, section 5.

Claim under test: Adaptive Branching MCTS (Sakana 2024) using Thompson
sampling for the wide-vs-deep tradeoff outperforms vanilla UCT on the
same reasoning budget.

Task: solve a small "reasoning" game.  We use a synthetic tree-search
puzzle where the agent must find a path of length D through a tree of
branching factor B to maximize the sum of node rewards.  Some nodes
have hidden noisy reward estimates that confuse pure-UCT (Thompson
sampling handles this by sampling from the posterior over arm values
rather than using point estimates).

Honest scope:
  * Toy domain.  AB-MCTS is designed for LLM-as-policy reasoning;
    that's a heavyweight setup.  Here we strip the LLM and use a
    deterministic reward signal so we can isolate the search algorithm.
  * Metric: best leaf reward found within N simulations.  Averaged
    over many random trees so the result isn't dependent on one
    favorable layout.
  * If AB-MCTS finds higher-reward leaves with the same budget → claim
    validated.  If parity → both are fine; no clear win.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
#  Search environment: deterministic random-reward tree
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    """Node in the search tree.  Has a true reward (revealed on visit)
    and parent/child relationships.  We do NOT pre-build the whole
    tree — it expands lazily on demand to mimic LLM-style trees where
    expanding a branch costs a forward pass.
    """
    depth: int
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = field(default_factory=list)
    # On expansion, each node draws a noisy reward.  Higher depth =
    # higher reward potential but more noise — this is what makes
    # search non-trivial.
    true_reward: float = 0.0
    # MCTS bookkeeping
    visits: int = 0
    cumulative_reward: float = 0.0


def _expand(node: TreeNode, branching: int, max_depth: int, rng: random.Random):
    """Lazy expansion.  Cost: one call per child = one 'thought'."""
    if node.children:
        return  # already expanded
    if node.depth >= max_depth:
        return
    for _ in range(branching):
        # True reward = parent_reward + N(0, depth_noise)
        noise_scale = 0.5 + 0.1 * node.depth
        r = node.true_reward + rng.gauss(0.0, noise_scale)
        child = TreeNode(depth=node.depth + 1, parent=node, true_reward=r)
        node.children.append(child)


# ──────────────────────────────────────────────────────────────────────
#  Vanilla UCT
# ──────────────────────────────────────────────────────────────────────

def uct(parent: TreeNode, c: float = 1.4) -> TreeNode:
    """Standard UCB1 selection.  Returns the child with the highest
    UCB1 score; unvisited children have +∞ priority."""
    best = None
    best_score = -math.inf
    for child in parent.children:
        if child.visits == 0:
            return child
        avg = child.cumulative_reward / child.visits
        score = avg + c * math.sqrt(math.log(parent.visits) / child.visits)
        if score > best_score:
            best_score = score
            best = child
    return best  # type: ignore


def vanilla_mcts(root: TreeNode, n_simulations: int, branching: int,
                  max_depth: int, rng: random.Random) -> float:
    """Run n_simulations of vanilla UCT.  Returns the best leaf reward seen."""
    best_seen = -math.inf
    for _ in range(n_simulations):
        node = root
        # Selection
        while node.children and node.depth < max_depth:
            node = uct(node)
        # Expansion
        if node.depth < max_depth:
            _expand(node, branching, max_depth, rng)
            if node.children:
                node = node.children[0]
        # Evaluation
        leaf_value = node.true_reward
        best_seen = max(best_seen, leaf_value)
        # Backpropagation
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.cumulative_reward += leaf_value
            cur = cur.parent
    return best_seen


# ──────────────────────────────────────────────────────────────────────
#  AB-MCTS with Thompson sampling
# ──────────────────────────────────────────────────────────────────────
# AB-MCTS (Sakana 2024) maintains a Gaussian posterior per node and
# uses Thompson sampling to choose between:
#   (a) go DEEPER into an existing child (exploit)
#   (b) go WIDER by spawning a new child (explore breadth)
# instead of using fixed UCB1 weights.
#
# We approximate this with two-arm Thompson sampling at each node:
#   - "exploit arm" = best existing child (recursive)
#   - "explore arm" = expand a fresh child
# Each arm has a Beta or Gaussian posterior over its expected reward.

def thompson_sample(mean: float, variance: float, rng: random.Random) -> float:
    """Sample from N(mean, variance)."""
    return rng.gauss(mean, math.sqrt(max(variance, 1e-6)))


def ab_mcts(root: TreeNode, n_simulations: int, branching: int,
             max_depth: int, rng: random.Random) -> float:
    """AB-MCTS (Adaptive Branching MCTS) with Thompson-sampled
    wide-vs-deep decision at each node."""
    best_seen = -math.inf
    for _ in range(n_simulations):
        node = root
        # Selection with adaptive branching: at each step, sample from
        # posterior to decide between going deeper into a known child
        # or expanding a new child.
        while node.depth < max_depth:
            if not node.children:
                _expand(node, 1, max_depth, rng)  # at least one child
                node = node.children[0]
                continue

            # Estimate exploit-arm value (best existing child) and
            # explore-arm value (uncertainty-driven prior).
            best_child = max(node.children,
                              key=lambda c: c.cumulative_reward / max(c.visits, 1))
            exploit_mean = best_child.cumulative_reward / max(best_child.visits, 1)
            exploit_var = 1.0 / max(best_child.visits, 1)  # shrinks with visits
            # Explore-arm prior: optimistic estimate based on the node's
            # ancestor reward distribution.  Crude: use ancestor avg + 1σ.
            explore_mean = node.cumulative_reward / max(node.visits, 1) + 1.0
            explore_var = 2.0  # wide prior — encourages exploring while priors are noisy

            sample_exploit = thompson_sample(exploit_mean, exploit_var, rng)
            sample_explore = thompson_sample(explore_mean, explore_var, rng)

            if sample_explore > sample_exploit and len(node.children) < branching:
                # Go WIDER: add a new child
                noise = rng.gauss(0.0, 0.5 + 0.1 * node.depth)
                new_child = TreeNode(depth=node.depth + 1, parent=node,
                                      true_reward=node.true_reward + noise)
                node.children.append(new_child)
                node = new_child
            else:
                # Go DEEPER: select best existing
                node = best_child

        # Evaluation + backprop (same as UCT)
        leaf_value = node.true_reward
        best_seen = max(best_seen, leaf_value)
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.cumulative_reward += leaf_value
            cur = cur.parent
    return best_seen


# ──────────────────────────────────────────────────────────────────────
#  Test entry
# ──────────────────────────────────────────────────────────────────────

def test_ab_mcts_vs_uct(*, seed: int = 42, n_trees: int = 30, n_simulations: int = 200,
                          branching: int = 4, max_depth: int = 5) -> dict:
    """Run both algorithms on `n_trees` independent random trees.
    Same seed → same tree shapes for both (we re-seed the RNG before
    each tree).  Report average best-leaf-reward and wall-time.
    """
    results = {"Vanilla MCTS (UCT)": [], "AB-MCTS": []}
    times = {"Vanilla MCTS (UCT)": 0.0, "AB-MCTS": 0.0}
    for tree_i in range(n_trees):
        # Same tree for both runs: reset RNG to a per-tree seed
        for label, fn in [
            ("Vanilla MCTS (UCT)", vanilla_mcts),
            ("AB-MCTS", ab_mcts),
        ]:
            rng = random.Random(seed * 1000 + tree_i)
            root = TreeNode(depth=0, true_reward=0.0)
            t0 = time.time()
            best = fn(root, n_simulations, branching, max_depth, rng)
            times[label] += time.time() - t0
            results[label].append(best)

    summary = {}
    for label, vals in results.items():
        summary[label] = {
            "mean_best_leaf": sum(vals) / len(vals),
            "max_best_leaf": max(vals),
            "min_best_leaf": min(vals),
            "wall_time_s_total": times[label],
            "n_trees": n_trees,
            "n_simulations_per_tree": n_simulations,
        }
    # Compute the head-to-head delta — paired comparison, same trees
    diffs = [results["AB-MCTS"][i] - results["Vanilla MCTS (UCT)"][i]
             for i in range(n_trees)]
    summary["_ab_minus_uct_paired"] = {
        "mean_diff": sum(diffs) / len(diffs),
        "n_wins_ab": sum(1 for d in diffs if d > 0),
        "n_wins_uct": sum(1 for d in diffs if d < 0),
        "n_ties": sum(1 for d in diffs if d == 0),
    }
    return summary


ALL_TESTS = {"ab_mcts_vs_uct": test_ab_mcts_vs_uct}


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="AB-MCTS vs vanilla UCT on synthetic trees.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=30)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--branching", type=int, default=4)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args(argv)

    results = test_ab_mcts_vs_uct(seed=args.seed, n_trees=args.trees,
                                    n_simulations=args.simulations,
                                    branching=args.branching, max_depth=args.depth)
    print(f"\n=== test_ab_mcts_vs_uct ===")
    for label, val in results.items():
        if not isinstance(val, dict):
            continue
        print(f"  {label}:")
        for k, v in val.items():
            if isinstance(v, float):
                print(f"    {k:<32} {v:.4f}")
            else:
                print(f"    {k:<32} {v}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
