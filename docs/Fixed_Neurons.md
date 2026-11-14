# Fixed neurons (`dnn_fixed.py`)

This note adds *fixed neurons* to the networks of
`docs/DiscreteNeuralNetworkNotes151126.md` (the *notes*) and proves that
the Theorem of the notes still holds. A fixed neuron has fixed weights and
exists before the first training round. Past shifts, the start clock and
twins are all fixed neurons. Nothing in RUN or TRAIN changes, and `dnn.py` is
used as it is.

Lemma, Theorem and section numbers without a prefix refer to the notes.

## 1 Why

In the notes every hidden neuron is silent unless the run starts from a key
(Lemma 1), so a network started on unseen data computes with its visible
neurons alone. A helper chain is a set of *future* shifts of a neuron,
$g(t+1), g(t+2), \dots$, and its values have to come from the key. A *past*
shift, $g(t-1), g(t-2), \dots$, can be computed during the run: copy $g$, then
copy the copy. It needs no key, so it also works on unseen data.

The causal variant (`docs/Causal_Variant.md`) adds such neurons during
training and loses the no-forgetting guarantee. Here they are installed before
round 1, and the guarantee is kept (Proposition 1).

## 2 Definitions

**Definition F1 (fixed neuron).** The visible neurons are split into *task*
neurons $T$ (which include the input neurons $I$) and *fixed* neurons $\Phi$.
Each fixed neuron $m$ is a visible non-input neuron with a weight vector $w_m$
that has **integer** entries on the visible neurons. The *initial network*
$N_0$ has no hidden neurons and

$$F(m) = \{w_m\} \ \text{ for } m \in \Phi, \qquad F(i) = \mathbb{R}^{n_0} \ \text{ for } i \in T,$$

where $n_0 = |T| + |\Phi|$. A single point is a box, and condition (D) holds
because there are no hidden neurons.

**Definition F2 (extension).** A *task sequence* is a sequence
$u = [u(0), \dots, u(k)]$ of activation vectors on $T$. Its extension
$E(u) = [v(0), \dots, v(k)]$ on all visible neurons is

$$v(t)_i = u(t)_i \ (i \in T), \qquad v(0)_m = 0, \qquad v(t+1)_m = \sigma(w_m \cdot v(t)) \ (m \in \Phi).$$

So every fixed neuron starts at 0 and then follows its own rule, reading the
sequence's values. This is well defined by induction on $t$; fixed neurons may
read each other and themselves.

**Definition F3 (training and running).** A round trains the task sequences
$V$ by calling TRAIN on $\{E(u) : u \in V\}$. The key of $u$ is the key of
$E(u)$. A run *without a key* starts from a state in which every hidden neuron
is 0.

### 2.1 The fixed neurons used here

Let $S \subset T$ be the neurons to be remembered and $d$ the depth. An
*always-on input* is an input neuron `on` that is 1 at every step.

| kind | neurons | weights | value on an extended sequence |
|---|---|---|---|
| past shift | $x_{i,s}$ for $i \in S$, $1 \le s \le d$ | $x_{i,1}$: 1 from $i$. $x_{i,s}$: 1 from $x_{i,s-1}$ | $u(t-s)_i$ for $t \ge s$, else 0 |
| start clock | $c_s$, $1 \le s \le d$ | $c_1$: 1 from `on`. $c_s$: 1 from $c_{s-1}$ | 1 for $t \ge s$, else 0 |
| twin | $z_{i,s}$ for $i \in S$, $1 \le s \le d$ | $z_{i,1}$: 1 from `on`, −1 from $i$. $z_{i,s}$: 1 from $z_{i,s-1}$ | $1 - u(t-s)_i$ for $t \ge s$, else 0 |

The last column follows from Definition F2 by induction on $s$. For example
$z_{i,1}(t+1) = \sigma(1 - u(t)_i) = 1 - u(t)_i$, and a copy delays its source
by one step. A twin is active exactly when its shift neuron is inactive, once
the stream has run for $s$ steps:

$$x_{i,s}(t) + z_{i,s}(t) = c_s(t) \quad \text{for all } t. \tag{1}$$

A twin needs the always-on input, because no neuron can fire from an all-zero
state (section 8.5).

## 3 Results

**Proposition 1 (the Theorem holds).** Start from $N_0$ and train any number
of rounds as in Definition F3. Then, in every round:

1. FEAS($m$) holds whenever TRAIN tests it for a fixed neuron $m$. So no
   helper is ever created for a fixed neuron, and a round adds at most
   $2k\,(|T| - |I|)$ neurons.
2. After the round, $F(m) = \{w_m\} \times \prod_h J_h$ for every fixed $m$,
   where $h$ ranges over the hidden neurons and each $J_h$ is an interval. The
   weights of a fixed neuron from the visible neurons never change.
3. All four parts of the Theorem hold for the extended sequences: learning, no
   forgetting, condition (D) and storage in $2n^2 + n$ numbers.
4. In particular, for every weight matrix whose rows lie in the boxes, every
   task sequence of every round so far is reproduced exactly on the task
   neurons when run from its zero-padded key.

*Proof.* (1) and (2), by induction over rounds. Assume (2) holds before round
$r$ (at the start there are no hidden neurons). During step 1 of TRAIN the box
of a fixed neuron $m$ is

$$\tilde F(m) = \{w_m\} \times \prod_{h \text{ old}} J_h \times \mathbb{R}^{\text{new}},$$

because ADD lifts the box of a visible neuron by $\mathbb{R}$. Let $a$ equal
$w_m$ on the visible neurons, any point of $J_h$ on each old hidden neuron, and
0 on each new one. Take a constraint pair $(x, y)$ of $m$, so $x = v'(t)$ and
$y = v(t+1)_m$ for some extended sequence. The old hidden neurons are 0 in
$x$ (step 0 of TRAIN), so $a \cdot x = w_m \cdot v(t)$, which is an integer.
By Definition F2, $y = \sigma(w_m \cdot v(t))$. If $y = 1$ then
$a \cdot x \ge 1$. If $y = 0$ then $a \cdot x < 1$, hence
$a \cdot x \le 0 \le 1 - \varepsilon$. So $a \in P_m \cap \tilde F(m)$ and
FEAS($m$) holds, at every point of the round. Step 1 therefore skips $m$. In step 2, SUBBOX($m$) returns a
box inside $\tilde F(m)$, whose visible coordinates are the point $w_m$. This
gives (2) after the round. The bound in (1) is Lemma 4, whose proof counts two
chains per visible non-input neuron that needs help; fixed neurons never do.

(3) The proofs of Lemmas 1–4 and of the Theorem use the starting network only
through condition (D) and the fact that every box is non-empty. They never use
$F(i) = \mathbb{R}^n$ for a visible neuron: Lemma 2 is applied to a hidden
neuron of the current round, whose box contains 0 whatever the visible boxes
are, and Lemma 3 takes any finite point of the box. $N_0$ satisfies (D) and its
boxes are non-empty, and the extended sequences are sequences on the visible
neurons, which is what TRAIN takes. So the Theorem applies to them as written.

(4) follows from (3), since $E(u)$ restricted to $T$ is $u$. $\square$

**Remark (skipping the fixed neurons).** The proof shows that the row equal
to $w_m$ on the visible neurons and 0 on every hidden neuron always lies in
$P_m \cap \tilde F(m)$. Using that row as the box of $m$ is therefore a valid
outcome of step 2, and it amounts to lifting $F(m)$ by $\{0\}$ for every new
neuron and never testing or refitting $m$. `dnn_fixed.train` does this by
default, which saves two linear programs per fixed neuron per round. Every
other neuron gets the same box either way, since its own linear programs do
not involve $F(m)$ (tested, and experiment 9 gives identical results).

**Proposition 2 (fixed neurons follow their rule on any run).** After any
number of rounds, let $A$ be any weight matrix whose rows lie in the boxes, and
take any run without a key: any visible starting state and any input stream.
Then for every $t$ and every fixed neuron $m$

$$v(t+1)_m = \sigma\bigl(w_m \cdot v(t)\bigr),$$

where the sum is over the visible neurons.

*Proof.* Every cohort of hidden neurons is 0 at time 0, so by Lemma 1 it is 0
at every step. By Proposition 1(2) the row of $m$ in $A$ equals $w_m$ on the
visible neurons, so $(A v(t))_m = w_m \cdot v(t)$. $\square$

**Corollary 1 (shifts work on unseen data).** In a run without a key in which
the fixed neurons start at 0 and `on` is always on, the identities of section
2.1 hold with $u$ replaced by the run's own values: $x_{i,s}(t) = v(t-s)_i$
for $t \ge s$ and 0 otherwise, and likewise for the clock, the twins and
equation (1). For an output neuron $i$, $v(t-s)_i$ is what the network actually
produced, right or wrong.

**Corollary 2 (recall without a key).** If a round adds no hidden neuron, the
key of each of its sequences $u$ is $u(0)$ on the task neurons and 0
everywhere else. Such a sequence is replayed from its first task state alone,
and by Proposition 1 this stays true after every later round.

**Proposition 3 (twins and the clock give the same learnable class).** Fix
$S \ne \emptyset$ and $d$. Compare the *clock layout* (shifts and clock) with
the *twin layout* (shifts and twins), both with the always-on input. For a
task neuron $g$ and any set of task sequences, the polyhedron $P_g$ of section
4.1 is non-empty in one layout exactly when it is non-empty in the other. So,
starting from an untrained network, the first round adds no hidden neuron in
one layout exactly when it adds none in the other.

*Proof.* Both layouts extend a task sequence step by step, so their constraint
pairs correspond one to one, with the same targets $y$. By (1), a weight
vector of the twin layout with $a_{i,s}$ on $x_{i,s}$ and $b_{i,s}$ on
$z_{i,s}$ gives, on every extended state,

$$\sum_{i,s} a_{i,s}\,x_{i,s} + b_{i,s}\,z_{i,s} = \sum_{i,s} (a_{i,s} - b_{i,s})\,x_{i,s} + \sum_s \Bigl(\sum_i b_{i,s}\Bigr) c_s ,$$

which is a weighted sum of the clock layout. Conversely a weight $\gamma_s$ on
$c_s$ equals $\gamma_s$ on both $x_{i_0,s}$ and $z_{i_0,s}$ for any one
$i_0 \in S$. The weights on the task neurons are unchanged. So each layout can
produce exactly the sums the other can, and the same constraints are
satisfiable. In the first round every task neuron is tested in an unbounded
box, so it needs no helper exactly when its $P_g$ is non-empty, which gives
the last sentence. $\square$

**What Proposition 3 does not say.** The two layouts have different boxes: a
box in one is generally not a box in the other. Which box SUBBOX keeps, how
well it generalises from little data, and what happens in later rounds can
therefore differ, and in experiment 9 they do. When helpers are needed, the
chains can also differ in length, because a hidden neuron's weights from the
visible neurons are restricted in sign. The twin layout also costs $|S|\,d$
neurons against $d$ for the clock.

**Remark (the twin of a predictor).** For a helper chain the twin already
exists: the anti-predictor $q$ of section 4.3 is 1 exactly when the predictor
$p$ is 0, for $t < k$. Lemma 3 is the statement that a predictor together with
its twin always makes a visible neuron feasible.

**Remark (why before round 1).** A fixed neuron fires on old sequences, unlike
a helper cohort, which condition (D) keeps silent. Take one task neuron $g$,
train the sequence $1,1,1$, so $F(g) = [1, \infty)$, and only then add the
shift $x_{g,1}$. On the old sequence $x_{g,1}$ is $0,1,1$. If $F(g)$ is lifted
by $\mathbb{R}$, as for any new neuron, the weight $-5$ from $x_{g,1}$ is
allowed, and then $g$ receives $1 - 5 < 1$ at the second step: the sequence is
forgotten. The lift that would be safe depends on the old sequences, which are
not stored; the only lift that is safe for every history is $\{0\}$, and then
the new neuron is useless to the old ones. This is why the causal variant
forgets, and why its features are safe when they are installed before
round 1 (experiment 10).

## 4 What fixed neurons give and what they do not

- **A wider learnable class, without a key.** A task neuron needs no helper in
  round 1 exactly when its next value is a threshold function, at margin
  $\varepsilon$, of the extended state. With shifts of depth $d$ that is a
  function of the last $d + 1$ task states, where before it was the current
  state alone.
- **Shifts add memory, not new kinds of function.** A function that is not a
  threshold function of the window, such as an XOR of two of its entries, is
  out of reach of shifts, and so is anything that depends on the whole
  history, such as parity. Other fixed neurons reach these, but only with one
  step of latency: see `docs/Latency_And_State.md`.
- **A fixed cost.** The network has $|\Phi|$ more neurons from the start. The
  storage is still $2n^2 + n$ numbers, and $|\Phi|$ does not grow.
- **The depth is chosen in advance.**
- **The wrong-box problem remains.** A box kept after a small first round can
  exclude every correct weight vector (`experiments/RESULTS.md`, sections 8.2
  and 9.2). Fixed neurons add coordinates, which makes this more likely, not
  less.

## 5 Implementation and tests

`dnn_fixed.py` builds the layout (`make_layout`), the initial network
(`make_net`) and the extension (`extend`), and calls `dnn.train` on the
extended sequences (`train`). `tests/test_dnn_fixed.py` checks:

- the values in section 2.1 and equation (1);
- Proposition 1 on random networks with random fixed neurons over several
  rounds: replay of every sequence so far with sampled weight matrices, intact
  fixed boxes, condition (D);
- Proposition 2 on runs from random visible states with random inputs;
- Example 1 of the notes ($1,1,1,0$) with two shifts of the neuron: no hidden
  neuron, and replay without a key;
- Tomita 2 with shifts of depth 1 of the input and the output: no hidden
  neuron, exact on all 256 streams of length 8.

Results are in `experiments/RESULTS.md`, sections 9 and 10. The feature
library, the state neurons and their tests are described in
`docs/Latency_And_State.md`.
