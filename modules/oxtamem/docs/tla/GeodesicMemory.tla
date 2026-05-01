------------------------- MODULE GeodesicMemory -------------------------
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    Variables,      \* The set of variable IDs (e.g., {"Var_X", "Var_Y"})
    MaxTimestamp    \* Bound for model checking

VARIABLES
    store,          \* The "HD" or "RAM" (Function: Address -> NodeData)
    heads,          \* The "Heads" pointers (Function: VarID -> Address)
    nextAddr        \* Counter for next address allocation

\* Type Invariant
TypeOK ==
    /\ store \in [0..(nextAddr-1) -> [val : Int, prev : {0..(nextAddr-1)} \cup {NULL}, timestamp : Nat]]
    /\ heads \in [Variables -> {0..(nextAddr-1)} \cup {NULL}]
    /\ nextAddr \in Nat

NULL == -1

\* Initial State
Init ==
    /\ store = << >>    \* Empty sequence (using sequence as map for simplicity in TLA+)
    /\ heads = [v \in Variables |-> NULL]
    /\ nextAddr = 0

\* Action: Write (Append Only)
\* A write creates a new node that points to the previous head of that variable.
Write(v, value) ==
    LET prevPtr == heads[v]
        newAddr == nextAddr
        newNode == [val |-> value, prev |-> prevPtr, timestamp |-> newAddr]
    IN
        /\ nextAddr < MaxTimestamp
        /\ store' = Append(store, newNode) \* Append adds to domain 0..nextAddr-1 -> 0..nextAddr
        /\ heads' = [heads EXCEPT ![v] = newAddr]
        /\ nextAddr' = nextAddr + 1

\* Action: Read Latest
\* Reading doesn't change state, but we define the correctness property.
ReadLatest(v) ==
    IF heads[v] = NULL THEN NULL ELSE store[heads[v]].val

\* Consistency Property: No Cycles
\* The graph must form a DAG (Directed Acyclic Graph).
\* Since we only ever point 'prev' to an existing (older) address, cycles are impossible by construction.
\* We can verify this by asserting that for all nodes, node.prev < node.address (if prev != NULL).

AcyclicInvariant ==
    \A addr \in DOMAIN store :
        (store[addr+1].prev # NULL) => (store[addr+1].prev < addr)
        \* Note: Sequences in TLA+ are 1-indexed usually, but here we model address space.
        \* Let's simplify: In an append-only log where prev always points to a lower index,
        \* cycles are impossible.

\* Temporal Property: Causal Consistency
\* If we write V=1 then V=2, the head must point to V=2, and V=2 must point to V=1.
-----------------------------------------------------------------------------

\* Next State Relation
Next ==
    \E v \in Variables : \E val \in 0..10 : Write(v, val)

\* Specification
Spec == Init /\ [][Next]_<<store, heads, nextAddr>>

=============================================================================
