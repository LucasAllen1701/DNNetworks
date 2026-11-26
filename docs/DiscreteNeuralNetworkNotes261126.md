# Discrete Neural Network Notes — corrected version

Corrected version of `Neural_Network_Notes.pdf` (Lucas Allen, November 2026). This version is dated 26 November 2026; it replaces the version of 25 November 2026 (`docs/DiscreteNeuralNetworkNotes251126.md`, kept unchanged, as is the version of 15 November). Math is written in LaTeX notation.

## 0 What changed

**Since the 25 November version.**

- **Smaller margin.** $\varepsilon = 10^{-4}$ instead of $1/2$ (section 4.1), with what it changes and why it should not go much lower.
- **Size.** A round returns a network of the same size or larger: it adds neurons only when some FEAS fails (section 4.4).
- **No cohort label in the network.** A network is now $(n, n_0, I, O, F)$, where the first $n_0$ neurons are the visible ones. The cohort function $c$ is defined separately (Definition 5) and is used in the proofs; WAKE only needs, during a round, the set of neurons it has created in that round (section 4.3).
- **Keys** have a section of their own (4.5): what is in one, what it does, where it is kept, and what sleep does to keys.
- **Training has two phases**: phase 1, wake (memorisation, expansion), is the TRAIN algorithm of the earlier versions (4.1–4.6); phase 2, sleep (generalisation, compression), is the rebuild-and-prune algorithm of experiment 15 (4.7), and 4.8 describes the cycle. Sections 1.3, 1.6, 5, 8 and 9 are updated to match.

**Since the 15 November version.** Section 3 (running a network) is rewritten to match `dnn.py`: how $A$ is chosen when a box is unbounded, the three kinds of initial vector, a finite number of steps, the floating-point threshold, reading outputs at depth $h$, what a run does and does not guarantee, and a worked example. Section 8, item 6, now describes the tolerance the code actually uses. Nothing else changed.

**Since the original notes:**

| Original | Corrected |
|---|---|
| $F(j)$ is any convex set | $F(j)$ is a **box** (product of intervals): $2n$ numbers per neuron, $2n^2$ in total, independent of how many sequences have been trained |
| Row/column mix-up when building $A$ | Row $i$ of $A$ is a point of $F(i)$ |
| Input clamping outside the time loop | Clamping inside the loop; $T_0$ is emitted |
| Strict constraint $(Av)_i < 1$ | Margin $\le 1-\varepsilon$, so every test is a linear program |
| Step 1 returns after the first feasible neuron | All non-input neurons are processed, using a worklist |
| Constraints imposed on input neurons | Input neurons are skipped |
| One new neuron, assumed feasible | Chains of predictor neurons, with a termination proof |
| New weight restricted to $[0,1]$; non-emptiness unproved | Unrestricted; an inhibitory *anti-predictor* covers the cases where a predictor cannot work; non-emptiness proved |
| $F$ not lifted to $\mathbb{R}^{n+1}$ | Explicit lift rule |
| Interference with old sequences left open | A sign condition on boxes gives a no-forgetting theorem with no stored sequences |
| Output is the network only | Output also includes each new sequence's extended initial vector |

## 1 Introduction

These neural networks learn sequences of activation vectors, in which every neuron is on or off. This section says where the design comes from, what it is for, and where the rest of the work is written down.

### 1.1 Two ideas

**Training by linear programming.** Take an ordinary network of threshold neurons. Finding the weights into one neuron is a linear program: every activation vector the network should respond to gives one linear constraint on those weights (section 4.1). If the program is infeasible, a neuron is added to make it feasible (section 4.3).

The difficulty is that the program needs every activation vector the network is meant to learn, including those from earlier training. So a neuron does not store its weights. It stores the *feasible region* of its weights, in the hope that the region carries what the earlier vectors had to say. Storing that region exactly takes more than polynomial space, so it is stored as an inner approximation, a box (section 5).

**Running the network repeatedly.** A network is not run once on one activation vector. It is run step after step, and each step depends on the activations of the step before (section 3). This gives the network a short-term memory. It also means that what is trained is a sequence of activation vectors and not a single one: a stream of inputs goes in and a stream of outputs comes out.

### 1.2 Depth

A layered network with $h$ layers needs $h$ steps to carry an input through to the output. So the number of steps $h$ between an input and the output that answers it is the depth of the computation. Nothing in RUN or WAKE fixes $h$: the training sequences do, by the step at which each output appears.

- $h = 1$ is a single layer. As with a single-layer perceptron, some tasks are then impossible whatever neurons are added (`docs/Latency_And_State.md`, Proposition 4).
- $h = 2$ is one hidden layer. It is enough for every task with finitely many states, but it can need one hidden neuron for each state and each pattern of inputs (Proposition 5).
- A larger $h$ leaves room for fewer and more general hidden neurons, provided something builds the layers in between. Nothing here does so yet.

### 1.3 Which neuron to add

When a linear program is infeasible, the choice that remains open is what the activations of the added neuron should be.

These notes add *predictors*, which copy the answer from one step in the future (section 4.3). That always works, and nothing learned earlier is lost (the Theorem of section 4.5). But a predictor's activations cannot be computed during a run. They have to be supplied at the start, as a *key*, and a network that needs a key for a sequence has memorised it.

A neuron whose activations can be computed from the past needs no key, and what it learns carries over to sequences that were never trained. The companion documents study such neurons:

- `docs/Fixed_Neurons.md`: neurons that remember the last few steps, installed before training;
- `docs/Latency_And_State.md`: the limit at $h = 1$, and state neurons, which track the state of a task's machine;
- `docs/State_Merging.md`: finding the state neurons from the training sequences;
- `docs/Box_Rule.md`: what the box costs when training is done in small rounds, and how keeping part of the exact region avoids it;
- `docs/Causal_Variant.md`: an earlier variant that adds such neurons during training and gives up the no-forgetting guarantee;
- `experiments/RESULTS.md`: everything that was measured.

Section 4 therefore splits training into two phases. In the **wake** phase new sequences are learned exactly by adding predictors, and nothing is forgotten. In the **sleep** phase the network is rebuilt from a buffer of the sequences seen, with neurons computed from the past in place of predictors wherever the data allow, and neurons that nothing needs are deleted.

### 1.4 Related work

The first idea has relatives in the constructive algorithms of 1989–90: the tiling algorithm (Mézard and Nadal, 1989), the upstart algorithm (Frean, 1990) and cascade-correlation (Fahlman and Lebiere, 1990). Each builds a network by adding units when the existing ones cannot fit the data. In the upstart algorithm an added unit corrects a unit that is wrongly on or wrongly off, much as the anti-predictor and the predictor do here, with the difference that it is computed from the inputs.

What these notes add is the stored feasible region, which lets training continue in rounds without the earlier data, and the second idea: sequences and short-term memory.

### 1.5 Goals

The long-term goals of the project are three programs that run on a local machine and are built from discrete neural networks:

1. one that plays Doom;
2. one that writes Lean 4 proofs: given a goal and its local context, it constructs a proof, either with tactics or as a proof term;
3. a chat bot.

These are a long way from what has been shown. The networks in these notes and the companion documents have been tested on tasks with one or two input neurons, a handful of states and exact data. The goals need many input neurons, data that are not exact, and far larger networks.

### 1.6 Notation and layout

$[n] = \{1,\dots,n\}$ and time is $t \in \{0,1,2,\dots\}$. Section 2 defines the networks, section 3 runs them and section 4 trains them (wake, 4.1–4.6; sleep, 4.7–4.8). Section 5 is the space bound, sections 6 and 7 answer the questions of the original notes and give examples, and section 8 lists limitations.

**References for 1.4.** M. Mézard and J.-P. Nadal (1989), Learning in feedforward layered networks: the tiling algorithm, *Journal of Physics A* 22. M. Frean (1990), The upstart algorithm: a method for constructing and training feedforward neural networks, *Neural Computation* 2. S. Fahlman and C. Lebiere (1990), The cascade-correlation learning architecture, *Advances in Neural Information Processing Systems* 2.

## 2 Definitions

**Definition 1 (activation).** $\sigma(x) = 0$ if $x < 1$ and $\sigma(x) = 1$ if $x \ge 1$, applied componentwise to vectors.

**Definition 2 (box).** A box in $\mathbb{R}^n$ is a non-empty set $\prod_{j=1}^n [l_j, u_j]$ with $l_j \in \mathbb{R}\cup\{-\infty\}$, $u_j \in \mathbb{R}\cup\{+\infty\}$, $l_j \le u_j$ (an infinite endpoint means that side is unbounded). $\mathbb{R}^n$ is a box.

**Definition 3 (network).** A discrete neural network is a tuple $(n, n_0, I, O, F)$ where

- $n$ is the number of neurons;
- $n_0 \le n$ is the number of *visible* neurons: neurons $1, \dots, n_0$ are the original network, and neurons $n_0+1, \dots, n$ are *hidden* neurons added by training, numbered in the order they were created;
- $I, O \subset [n_0]$ are disjoint sets of visible neurons, the input and output neurons;
- $F(i) = \prod_j [l_{ij}, u_{ij}]$ is a box in $\mathbb{R}^n$: the allowed weight vectors *into* neuron $i$.

An untrained network has $n = n_0$ and $F(i) = \mathbb{R}^n$ for all $i$.

**Definition 4 (activations).** $v(t) \in \{0,1\}^n$ is the activation vector at time $t$.

**Definition 5 (cohort).** Training happens in rounds $r = 1, 2, \dots$ (section 4). The *cohort function* $c : [n] \to \{0,1,2,\dots\}$ gives $c(i) = 0$ if $i$ is visible and $c(i) = r$ if $i$ is a hidden neuron created in round $r$. The neurons with the same value of $c$ form a *cohort*. Because hidden neurons are numbered in order of creation, $c$ is non-decreasing.

$c$ is **not part of the network**. It is a record of the training history, used to state condition (D) and in the proofs. Training never needs to look up $c$ for an earlier round: during round $r$ it only needs to know which neurons it has created so far in that round (section 4.3), and the boxes it sets up then carry everything later rounds need. (Earlier versions stored $c$ as part of the network, and `dnn.py` still does, as `Net.c`; nothing in it reads $c$ outside the round that sets it.)

**Dormancy condition (D).** For every hidden neuron $i$ and every neuron $j$ with $c(j) \ne c(i)$: $u_{ij} \le 0$. In words, a hidden neuron can only be excited by its own cohort. Visible neurons count as cohort $0$, so a hidden neuron is never excited by a visible one.

## 3 Running a network

A network $(n, I, O, F, c)$ stores boxes, not weights. To run it, a weight matrix is first chosen from the boxes (3.1) and an initial vector is chosen (3.2). RUN (3.3) then steps the network forward. The code is `dnn.run` and `dnn.sample_weights`.

### 3.1 Choosing the weight matrix

Build $A \in \mathbb{R}^{n\times n}$ by choosing, for each $i$, a point $a_i \in F(i)$ as the $i$-th **row** of $A$. So $A_{ij}$ is the weight of the synapse from neuron $j$ to neuron $i$, and $A_{ii}$ is a self-connection.

- Every entry of $A$ must be a real number, but a box may have infinite endpoints. In the code, an unbounded side is cut off a distance $s$ from the finite side ($s = 5$), and a coordinate unbounded on both sides is taken from $[-s/2, s/2]$. Any finite choice inside $F(i)$ is allowed; the cut-off only makes sampling possible.
- The code offers the lower corner $a_i = (l_{ij})_j$, the upper corner $a_i = (u_{ij})_j$, a uniformly random point, and (in `dnn_buffer`) the midpoint. Tests replay trained sequences with several of these, including both corners.
- Rows of input neurons are computed in step 2.1 below but always overwritten in step 2.2, so their boxes do not matter.

$A$ is chosen once and kept fixed for the whole run.

### 3.2 The initial vector

The only state a network carries from one step to the next is its activation vector. So $v(0) \in \{0,1\}^n$ decides which of its learned behaviours a run follows. There are three cases.

1. **A sequence of the latest round.** $v(0)$ is its key $v'(0)$, returned by WAKE (sections 4.4 and 4.5). It contains the sequence's first activation vector on the visible neurons and the starting values of the helper neurons of that round.
2. **A sequence of an earlier round.** $v(0)$ is its key, padded with zeros on every neuron created after that round.
3. **A stream that was never trained.** There is no key. $v(0)$ is the stream's first vector on the visible neurons, with every hidden neuron (and every fixed neuron, section 4.7) at $0$ (`dnn_fixed.start`). Neurons that need a key, such as predictor chains, then stay silent (Lemma 1), and only neurons computed from the past can respond. This is the case in which generalisation is measured.

In every case the entries of $v(0)$ on input neurons are overwritten by $S_0$ in step 1 of RUN.

### 3.3 The algorithm

**ALGORITHM RUN**

INPUTS: $A$; an initial vector $v(0) \in \{0,1\}^n$; a number of steps $k \ge 0$; sets $S_0, S_1, \dots, S_k \subset I$ ($S_t$ is the set of input neurons that are on at time $t$).

OUTPUT: the activation vectors $v(0), v(1), \dots, v(k)$ on all $n$ neurons, and the sets $T_t = \{o \in O : v(t)_o = 1\}$ for $t = 0, \dots, k$.

STEPS:

1. For each $i \in I$ set $v(0)_i := 1$ if $i \in S_0$, else $0$. Emit $T_0$.
2. For $t = 0, 1, \dots, k-1$:
   1. $w := \sigma(A\,v(t))$, computed for every neuron, input neurons included.
   2. For each $i \in I$ set $w_i := 1$ if $i \in S_{t+1}$, else $0$.
   3. $v(t+1) := w$. Emit $T_{t+1}$.

Every non-input neuron at time $t+1$ is computed only from $v(t)$, all at once. A neuron sees its own previous value through $A_{ii}$, and sees the inputs of time $t$ (already in $v(t)$), not those of time $t+1$.

**Floating point.** Definition 1 fires a neuron when $x \ge 1$. The boxes come from linear programs, whose solutions can miss the threshold by rounding error, so the code fires when $x \ge 1 - \delta$ with $\delta = 10^{-7}$. Constraints with $y = 0$ have margin $\varepsilon$ ($10^{-4}$ in this version, $1/2$ in experiments 1–36), still a thousand times $\delta$, so the tolerance cannot make a silent neuron fire.

### 3.4 Reading the outputs

An input at time $t$ reaches the neurons of the first layer at $t+1$ and an output at $t+h$, where $h$ is the depth (section 1.2). So $T_{t+h}$ is the answer to $S_t$. The outputs $T_0, \dots, T_{h-1}$ answer no input: for a trained sequence they come from the key, and for an untrained stream they depend on the $0$ entries of $v(0)$. When a stream is scored, its outputs are compared from time $h$ on. A task that must answer every input needs at least $h$ steps between them; the "wait" games of the experiments hold each screen for several steps for this reason.

### 3.5 What a run guarantees

- **Trained sequences.** By the Theorem of section 4.6, every sequence of every round, run from its (padded) key with its own input sets, is reproduced exactly on all neurons, for every $A$ chosen as in 3.1.
- **Untrained streams.** Nothing is guaranteed. Different choices of $A$ inside the same boxes can give different runs, since the boxes only constrain $A$ on the activation vectors seen in training.
- **Determinism.** For a fixed $A$, $v(0)$ and $S_0, \dots, S_k$, the run is deterministic.

### 3.6 Example

Take Example 1 of section 7: one visible neuron, no inputs, trained on the sequence $1,1,1,0$. Training gives 3 neurons, the key $(1,1,1)$, and the rows $a_1 = (0,1,0)$, $a_2 = (0,0,1)$, $a_3 = (0,0,0)$. RUN with $k = 3$ gives

$$v(0) = (1,1,1),\quad v(1) = (1,1,0),\quad v(2) = (1,0,0),\quad v(3) = (0,0,0).$$

Neuron 1 reads $1,1,1,0$, the trained sequence. Neurons 2 and 3 are the predictor chain: the key stores what neuron 1 must do, and the chain delivers it one step at a time. Run from $v(0) = (1,0,0)$ instead (case 3 of 3.2: no key), the same network gives $1,0,0,0$, because the chain is silent.

## 4 Training a network

Training has two phases, following Lucas's idea of expansion and compression.

- **Phase 1, wake (memorisation, expansion).** Training happens in rounds $r = 1, 2, \dots$. Each round takes a finite set of new sequences and learns them exactly, without forgetting earlier ones, adding neurons where needed (sections 4.1–4.6). The neurons it adds are predictors: their activations come from keys (section 4.5), so what they learn does not carry over to new sequences.
- **Phase 2, sleep (generalisation, compression).** From time to time the network is rebuilt from a buffer of the sequences seen. A finite machine is found from the buffer and installed as fixed neurons, which compute their values from the past; a fresh network is trained on the buffer by one wake round; and every neuron that nothing needs is deleted (section 4.7). Section 4.8 describes the cycle of the two phases.

The sleep algorithm of section 4.7 is the one tested in experiment 15 (`dnn_sleep.sleep`). Its machine-finding step was designed by Claude, and only works for finite machines over a few input patterns. What sleep should do in general is open (section 8).

### 4.1 Constraints for one neuron

Suppose the current network has $m$ neurons and each training sequence is $[v(0), \dots, v(k)]$ with $v(t) \in \{0,1\}^m$. For a neuron $g$, every sequence and every $t \in \{0,\dots,k-1\}$ gives a pair

$$x = v(t) \in \{0,1\}^m, \qquad y = v(t+1)_g \in \{0,1\}.$$

Fix a margin $\varepsilon \in (0,1]$. The weight vector $a \in \mathbb{R}^m$ into $g$ must satisfy

$$a \cdot x \ge 1 \ \text{ if } y = 1, \qquad a \cdot x \le 1 - \varepsilon \ \text{ if } y = 0.$$

Call the set of such $a$ the polyhedron $P_g$. (The margin replaces the strict inequality $<1$, which a linear program cannot express.)

**The value of $\varepsilon$.** This version uses $\varepsilon = 10^{-4}$ (Lucas's decision, 26 November). Every result in section 4 holds for any $\varepsilon \in (0,1]$.

- A smaller $\varepsilon$ rejects fewer problems. For finitely many pairs, the strict problem $a\cdot x < 1$ is feasible exactly when the problem with margin $\varepsilon$ is feasible for some small enough $\varepsilon$, so a small margin comes close to the original strict inequality. Fewer infeasible problems means fewer added neurons. On 150 random networks (section 9), $\varepsilon = 10^{-3}$ and $10^{-4}$ added 1062 hidden neurons in total, against 1135 at $\varepsilon = 1/2$, and every replay was exact.
- $\varepsilon$ should stay well above the firing tolerance $\delta = 10^{-7}$ of section 3.3 and the feasibility tolerance of the LP solver (also about $10^{-7}$). In practice it should not go below about $10^{-5}$.
- On trained vectors a smaller margin changes nothing: the boxes still reproduce every trained sequence. On untrained vectors, the weighted sums of silent neurons may now lie just below $1$, so vectors close to trained ones are more likely to make them fire. Whether that helps or hurts generalisation has not been measured.
- Experiments 1–36 used $\varepsilon = 1/2$, and `dnn.py`'s default is still $1/2$. Results at the new value will differ, so rows of earlier experiments are not expected to reproduce at $10^{-4}$.

### 4.2 Two subroutines on boxes

Because $x$ has entries in $\{0,1\}$, a box lies inside a constraint exactly when its worst corner does:

$$\prod_j [l_j,u_j] \subset \{a : a\cdot x \ge 1\} \iff \sum_{j : x_j = 1} l_j \ge 1, \qquad
\prod_j [l_j,u_j] \subset \{a : a\cdot x \le 1-\varepsilon\} \iff \sum_{j : x_j = 1} u_j \le 1-\varepsilon .$$

Both conditions are linear in the endpoints. This is why boxes are used: every step below is a linear program.

**FEAS($g$).** Decide whether $P_g \cap \tilde F(g) \neq \emptyset$, where $\tilde F(g) = \prod_j[\tilde l_j,\tilde u_j]$ is the current box of $g$. This is the LP feasibility problem: find $a$ with $\tilde l \le a \le \tilde u$ and the constraints of 4.1.

**SUBBOX($g$).** Return a box $F'(g) \subset P_g \cap \tilde F(g)$. With a width cap $\omega > 0$ (e.g. $\omega = 1$), solve

$$\max \sum_j w_j \quad \text{s.t.} \quad
\tilde l_j \le l'_j,\quad u'_j \le \tilde u_j,\quad 0 \le w_j \le \omega,\quad w_j \le u'_j - l'_j,$$
$$\sum_{j : x_j=1} l'_j \ge 1 \ \text{ for each pair with } y=1, \qquad \sum_{j : x_j=1} u'_j \le 1-\varepsilon \ \text{ for each pair with } y=0 .$$

(Bounds with infinite $\tilde l_j$ or $\tilde u_j$ are omitted.) Then *expand*: set $l'_j := \tilde l_j$ for every $j$ that appears in no pair with $y = 1$, and $u'_j := \tilde u_j$ for every $j$ that appears in no pair with $y = 0$. Expansion keeps all constraints true and only enlarges the box.

SUBBOX has a solution exactly when FEAS($g$) is true: a feasible point $a$ is the box $l' = u' = a$, and conversely the lower corner of any feasible box is a feasible point. Any other rule for picking a box inside $P_g \cap \tilde F(g)$ is also correct; the choice only affects how much freedom is left for later rounds.

### 4.3 Adding a neuron

Let $g$ be a neuron for which FEAS($g$) fails. Two kinds of helper neuron can be added. In each training sequence $[v(0),\dots,v(k)]$ their activations are

$$\textbf{predictor } p \text{ of } g: \quad p(t) = v(t+1)_g \ \ (t < k), \qquad p(k) = 0,$$
$$\textbf{anti-predictor } q \text{ of } g: \quad q(t) = 1 - v(t+1)_g \ \ (t < k), \qquad q(k) = 0.$$

The predictor fires one step before $g$ should fire and excites it. The anti-predictor fires one step before $g$ should stay silent and inhibits it.

During round $r$, WAKE keeps the set $N$ of neurons it has created so far in this round ($N = \emptyset$ when the round starts). The *older hidden neurons* are the hidden neurons that existed when the round started: those $j > n_0$ with $j \notin N$.

**ADD(column).** With $m$ neurons so far:

1. $m := m+1$, and append the column to every training sequence.
2. Lift every existing box by one coordinate (the weight from the new neuron):
   - if $j$ is visible or $j \in N$: $\tilde F(j) := \tilde F(j) \times \mathbb{R}$;
   - if $j$ is an older hidden neuron: $\tilde F(j) := \tilde F(j) \times (-\infty, 0]$.
3. The new neuron's box is $\tilde F(m) := \prod_{j=1}^m J_j$ with $J_j = \mathbb{R}$ if $j \in N$ or $j = m$, and $J_j = (-\infty, 0]$ otherwise (visible neurons and older hidden neurons).
4. $N := N \cup \{m\}$.

With $c(m) := r$, steps 2 and 3 are exactly what keeps condition (D) true. They are the only place where training needs to know which round a neuron belongs to, and they only need to tell the current round's neurons ($N$) from the rest. That is why $c$ need not be stored (Definition 5).

*Remark.* Using $\{0\}$ in place of $(-\infty, 0]$ in steps 2 and 3, so that weights between different rounds are exactly $0$, also keeps (D) and every result below. It removes the option of inhibition between rounds, which can only make FEAS fail more often. `dnn.py` uses $(-\infty, 0]$.

### 4.4 Phase 1: the wake round

**ALGORITHM WAKE (round $r$)** (called TRAIN in earlier versions; `dnn.train`)

INPUTS: a network $(n, n_0, I, O, F)$ satisfying (D); a finite set $V$ of sequences $[v(0),\dots,v(k)]$ of activation vectors **on the visible neurons** (lengths may differ; the input neurons' values are part of the sequences); a margin $\varepsilon \in (0,1]$.

OUTPUT: a network $(n', n_0, I, O, F')$ with $n' \ge n$, and for each sequence in $V$ its key $v'(0) \in \{0,1\}^{n'}$ (section 4.5). The network grows only if a neuron is added: $n' = n$ exactly when FEAS($g$) holds the first time it is checked, for every visible non-input neuron $g$. Even then the boxes of the visible non-input neurons are replaced by SUBBOX in step 2, so they may shrink.

STEPS:

0. Extend every sequence in $V$ by zeros on all hidden neurons. Set $\tilde F := F$, $m := n$, $N := \emptyset$. Let the worklist $W$ and the refit list $R$ both be the visible non-input neurons.
1. While $W$ is non-empty, remove a neuron $g$ from $W$.
   - If FEAS($g$) holds, continue.
   - (a) Otherwise ADD the predictor $p$ of $g$. If FEAS($g$) now holds, put $p$ in $W$ and $R$ and continue. (For a hidden $g$ this always succeeds, by Lemma 2.)
   - (b) Otherwise ($g$ is visible) undo the addition of $p$ and ADD the anti-predictor $q$ of $g$ instead. If FEAS($g$) now holds, put $q$ in $W$ and $R$ and continue.
   - (c) Otherwise undo the addition of $q$ and ADD both $p$ and $q$. FEAS($g$) now holds by Lemma 3. Put $p$ and $q$ in $W$ and $R$.
2. For each $g \in R$ set $F'(g) :=$ SUBBOX($g$). For every other neuron (input neurons and older hidden neurons) set $F'(g) := \tilde F(g)$.
3. If the sleep phase is used, add the sequences of $V$ to the buffer $B$ (section 4.7).
4. Return $(m, n_0, I, O, F')$ and, for each sequence, row $0$ of its extended form: its key.

### 4.5 Keys

**Definition 6 (extended sequence, key).** Let $[v(0),\dots,v(k)]$ be a sequence of $V$ in round $r$. WAKE extends it to all $n'$ neurons: $v'(t) = (v(t), z(t))$, where $z(t)$ holds the activations of the hidden neurons at time $t$. These are $0$ on the older hidden neurons (step 0), and on each neuron created in round $r$ they are the column ADD gave it, worked out from this sequence's own values. The **key** of the sequence is $v'(0)$, the extended vector at time $0$.

**What is in a key.**

- On the visible neurons, the key is $v(0)$, the sequence's own first vector.
- On older hidden neurons, it is $0$.
- On the round's helpers, it is their values at time $0$. For a predictor $p$ of $g$, $p(0) = v(1)_g$; for an anti-predictor, $q(0) = 1 - v(1)_g$. In a chain $h_1, h_2, \dots$ started for $g$, $h_s(0) = h_1(s-1) = v(s)_g$. So the key holds the *future* values of $g$, one per chain neuron. This is the part of the sequence the network could not compute from the past: what it has memorised.

A helper added because of one sequence of $V$ gets a column in every sequence of $V$ (each worked out from that sequence), so a sequence's key can have non-zero helper entries even when that sequence alone would have needed no helper.

**What a key does.** By the Theorem (section 4.6), RUN started from the key, with the sequence's input sets $S_t$, reproduces $v'(t)$ for every $t$ and for every weight matrix chosen from the boxes. The key is what selects which memorised sequence the network replays. Started instead from $(v(0), 0)$, with every hidden neuron at $0$, the round's neurons are $0$ at time $0$ and stay $0$ for ever (Lemma 1). The network then behaves as if the round had added no neurons, and its output depends only on what the visible neurons compute (example in section 3.6).

**Keys from earlier rounds.** A key from round $r$ is extended with zeros for every neuron created in a later round. By the Theorem it still replays its sequence.

**Where keys are kept.** Outside the network, by the caller: one vector per sequence. Only $v(0)$ and the entries for the sequence's own round can be non-zero, so a key costs at most $n_0$ plus the number of neurons that round added. The space bound of section 5 counts the network only, not the keys.

**Keys and generalisation.** A sequence that replays from $(v(0), 0)$ needs no key: the network computes it from its start. An untrained stream has no key, so it is always run this way (section 3.2, case 3). A network that needs a key for a sequence has memorised it, and the fraction of trained sequences that replay *without* their keys is a measure of how much has been generalised rather than memorised (used in experiment 15).

**Keys and sleep.** Sleep builds a new network, so every old key stops being valid. Sleep returns a new key for each sequence of the buffer, and its aim is that as many as possible of these are just $(v(0), 0)$ on the new network.

### 4.6 Correctness of the wake phase

**Lemma 1 (dormancy).** Assume (D). If every neuron of a cohort $C$ is $0$ at time $t$, then every neuron of $C$ is $0$ at time $t+1$, for every choice of $A$.

*Proof.* For $h \in C$ and $a \in F(h)$: $a \cdot v(t) = \sum_{j \notin C} a_j v(t)_j \le 0 < 1$, since $a_j \le 0$ for $j \notin C$ and $v(t)_j = 0$ for $j \in C$. $\square$

**Lemma 2 (predictor).** Suppose some $a \in \tilde F(g)$ satisfies every constraint of $g$ with $y = 0$. Then after adding the predictor of $g$, FEAS($g$) holds. In particular this applies whenever $0 \in \tilde F(g)$: to every neuron in round 1 whose box still contains $0$, and to every hidden neuron of the current round.

*Proof.* Take the new weight $b = \max(0,\, 1 - \min a\cdot x)$, the minimum over pairs with $y = 1$. On pairs with $y=1$ the predictor is $1$, so $a\cdot x + b \ge 1$. On pairs with $y = 0$ it is $0$, so the value is $a \cdot x \le 1-\varepsilon$. The new coordinate of $g$ is unrestricted, because $g$ is visible or in $N$. If $0 \in \tilde F(g)$, then $a = 0$ works since $0 \le 1 - \varepsilon$. $\square$

**Lemma 3 (predictor and anti-predictor).** Let $g$ be visible. After adding both $p$ and $q$, FEAS($g$) holds.

*Proof.* Take any finite $a \in \tilde F(g)$ and let $L = \min a\cdot x$, $U = \max a \cdot x$ over all pairs. Give $p$ the weight $b_p = \max(0, 1 - L)$ and $q$ the weight $b_q = \min(0, 1 - \varepsilon - U)$. On pairs with $y = 1$: $p = 1$, $q = 0$, value $a\cdot x + b_p \ge 1$. On pairs with $y=0$: $p = 0$, $q = 1$, value $a \cdot x + b_q \le 1-\varepsilon$. $\square$

**Lemma 4 (termination).** Let $k$ be the length of the longest sequence in $V$. Step 1 terminates, and the round adds at most $2k\,(n_0 - |I|)$ neurons.

*Proof.* Helpers of helpers are always predictors (Lemma 2), so each helper of a visible neuron starts a chain $h_1, h_2, \dots$ where $h_{s+1}$ is the predictor of $h_s$. Then $h_s(t) = h_1(t+s-1)$ when $t+s-1 \le k-1$ and $0$ otherwise. The targets of $h_k$ are therefore all $0$, so $a = 0$ is feasible and the chain stops with at most $k$ neurons. Each visible non-input neuron starts at most two chains. $\square$

**Theorem.** After any number of wake rounds, for every weight matrix $A$ whose rows lie in the boxes:

1. *(Learning)* every sequence of the latest round, run from its key with its inputs, is reproduced exactly on all neurons;
2. *(No forgetting)* every sequence of every earlier round, run from its zero-padded key, is still reproduced exactly;
3. condition (D) holds;
4. the network is stored in $2n^2$ numbers and $n_0$, where $n$ is the current number of neurons.

*Proof.* (3) holds by the lift rule in ADD, and SUBBOX only shrinks boxes. (4) is Definition 3.

(1) Induct on $t$. Input neurons are clamped to the sequence's values. For $g \in R$, $F'(g) \subset P_g$, so $\sigma(a_g \cdot v'(t)) = v'(t+1)_g$. Older hidden neurons are $0$ in the extended sequence and stay $0$ by Lemma 1.

(2) Let $[v(0),\dots,v(k)]$ be a sequence learned before round $r$, and $A'$ a matrix for the new network. Induct on $t$ with the claim that the state is $(v(t), 0)$. The new cohort is $0$ at time $t$, so it is $0$ at time $t+1$ by Lemma 1. For an old neuron $i$, the row $a'_i \in F'(i) \subset F(i) \times (\text{intervals})$ restricts to some $a_i \in F(i)$, and $a'_i \cdot (v(t), 0) = a_i \cdot v(t)$. So the old neurons behave as in the old network, which reproduces the sequence for every choice of rows in $F$. $\square$

### 4.7 Phase 2: sleep

Sleep needs two things the wake phase does not have: fixed neurons and a buffer.

**Fixed neurons** (`docs/Fixed_Neurons.md`, `dnn_fixed.py`). A fixed neuron is a visible, non-input neuron whose box on the visible neurons is a single point with integer entries; its weights from hidden neurons may be intervals. It is installed before round 1. Like any neuron, its value at $t+1$ is $\sigma$ of its weighted sum at $t$, so its value is computed from the past and it needs no key. The visible neurons are then of two kinds: *task neurons*, whose values the training sequences give (inputs, outputs and any others), and fixed neurons, whose values are worked out by running their rules along each sequence from $0$ (`dnn_fixed.extend`). With integer weights a silent fixed neuron has a weighted sum of at most $0 \le 1-\varepsilon$, so its constraints always hold for any $\varepsilon \le 1$. `docs/Fixed_Neurons.md` proves that the Theorem still holds and that fixed neurons compute their rules on any run started with the hidden neurons at $0$.

**The buffer** $B$ is a list of the sequences seen in the wake phase, on the task neurons only (step 3 of WAKE). It is stored data, kept outside the network; how large it may be and which sequences may leave it are discussed in 4.8.

**ALGORITHM SLEEP** (`dnn_sleep.sleep`)

INPUTS: a buffer $B$ of sequences on $n_{\text{task}}$ task neurons with inputs $I$ and outputs $O$; the depth $h$ (section 1.2); a margin $\varepsilon$.

OUTPUT: a new network $(n, n_0, I, O, F)$ and a key for each sequence of $B$.

STEPS:

1. **Find a machine** (`dnn_merge.learn_machine`, `docs/State_Merging.md`). Read each sequence of $B$ as a stream of input patterns, each answered by the output $h$ steps later. Build the *prefix tree*: one state for each input prefix seen, with the observed output on each edge; this machine has memorised $B$. Then merge states: taking states in order of prefix length, merge each into the first earlier state that the data do not contradict, where two states are contradicted when some continuation seen from both gives different outputs (the RPNI order of Oncina and García, 1992). The result is a partial machine $(\delta, \text{out})$. Give every state a transition for every input pattern, staying in the same state where none was seen.
2. **Install** (`dnn_merge.make_layout`, `dnn_fixed.add_automaton`, `docs/Latency_And_State.md`). Build a new untrained network whose visible neurons are the task neurons, an always-on input neuron, and fixed *state neurons* that track the state of the machine. $n_0$ counts all of them; there are no hidden neurons.
3. **Train.** Run one WAKE round of the new network on $B$, with each sequence extended by the fixed neurons' values. Sequences the machine explains need no helpers; any it does not explain are memorised with predictor chains and keys, as in any wake round.
4. **Prune** (`dnn_sleep.prune`). Repeat until nothing changes: delete every neuron $j$ that is neither a task neuron nor the always-on neuron and for which $l_{ij} \le 0 \le u_{ij}$ for every other remaining neuron $i$; remove its row, its column and its entries in the keys.
5. Return the network and the keys.

**Proposition (sleep).** Every sequence of $B$ is reproduced exactly by the network SLEEP returns, run from its new key, for every weight matrix whose rows lie in the boxes.

*Proof.* After step 3 this is the Theorem. In step 4, every weight leaving a deleted neuron may be $0$, and the boxes are products of intervals, so setting all those weights to $0$ gives a point of every box. By the Theorem every sequence still replays with those weights, and then no remaining neuron depends on the deleted ones, so the sequences with their columns removed replay on the smaller network. Deleting several neurons at once, and repeating, is the same argument. A fixed neuron's box is a point, so it is deletable only if every kept neuron that reads it does so with weight $0$; the kept fixed neurons therefore still compute their rules. $\square$

**What sleep does not guarantee.**

- Sequences that are not in $B$ are not kept. No forgetting holds only for the buffer, and every key from before the sleep stops being valid.
- Behaviour on untrained streams changes. This is the point of sleep: memorised neurons are replaced by state neurons that generalise when the machine is right.

**What it gave** (experiment 15; six stream tasks, 32 streams arriving 4 per round, sleep after every second round, a buffer of all streams). Sleep found the true machine in 59 of 60 runs by round 6, against 26 when the machine found in round 1 was kept. After a sleep the network needed no hidden neurons. Recall from keys was exact throughout. Without sleep, 36 hidden neurons had been added by round 8, and only 73% of the streams seen replayed without keys. Step 4 found nothing to delete in these tasks.

**Limits.** Step 1 is Claude's design, and it needs exact data and few input patterns (one transition per input pattern). For noisy data, the split search of experiment 36 (`dnn_noise.learn_machine_split`) can replace it. For screens, input patterns must first be reduced to letters (screen-region features, experiments 23, 30 and 33), which are found by a separate search, not by sleep. So this sleep compresses only into finite machines over small alphabets. A general sleep phase is an open problem (section 8).

### 4.8 The wake–sleep cycle

1. As new sequences arrive, train them in WAKE rounds and add them to $B$. Between sleeps everything is exact and nothing is forgotten (the Theorem).
2. Every few rounds (every second round in experiment 15), run SLEEP on $B$ and replace the network and the keys with the ones it returns.

Round 1 can itself be a sleep: building the first network from the first sequences by SLEEP, as experiment 15 does.

**Bounded buffers.** Lucas's decision 5 is a buffer of early data, for now. A buffer that keeps every sequence grows without bound. When it must shrink, which sequences may leave it matters. Keeping the most recent ones loses the sequences that ruled out wrong machines (experiment 15, buffer of 16: 3 runs ended with a wrong machine). Keeping the *witnesses*, the sequences whose removal would change the machine found, does better (experiment 18; for letters, experiment 25).

## 5 The space bound

**What is stored.** $2n$ interval endpoints per neuron, $2n^2$ numbers, and the number $n_0$ of visible neurons. This does not depend on the number or length of sequences trained. In the wake phase the training sequences of a round are needed only during that round. Two things are kept outside the network: the keys (section 4.5), and, if the sleep phase is used, the buffer $B$ (section 4.7). The buffer grows with the data unless it is bounded (section 4.8).

**Boxes are an inner approximation, and that is unavoidable.** The exact set of weights consistent with everything trained so far cannot, in general, be stored in polynomial space. Take $n$ even and let $M$ be the set of vectors in $\{0,1\}^n$ with exactly $n/2$ ones. For each subset $T \subset M$, the region $\{a : a\cdot x \ge 1 \ \forall x \in T\}$ is different: if $x_0 \in T \setminus T'$, the point with entries $2/n - \delta$ on the support of $x_0$ and a large value elsewhere satisfies every constraint of $T'$ but violates $x_0$. So there are $2^{|M|}$ distinct exact regions and any exact representation needs at least $\binom{n}{n/2} \approx 2^n/\sqrt{n}$ bits.

**What the approximation costs.** Every point of a box is a correct weight vector, so correctness is never lost. The cost is that a later sequence which fits the exact region may miss the box, in which case a helper neuron is added that an exact method would not need.

**Why boxes rather than ellipsoids.** An ellipsoid also takes polynomial space ($O(n^2)$ numbers per neuron), but: fitting one needs semidefinite programming rather than linear programming; it cannot represent an unbounded region such as $\mathbb{R}^n$, so weights would need an artificial bound; and neither the lift to $\mathbb{R}^{n+1}$ nor the sign condition (D) is an ellipsoid operation, while both are trivial for boxes.

**Number sizes.** The bound counts numbers. In floating point each is one machine word. In exact rational arithmetic the endpoints can grow in size over rounds; rounding lower endpoints up and upper endpoints down to a fixed precision keeps a box inside its feasible set as long as it stays non-empty.

## 6 Answers to the three open questions

1. *Is $(F(i) \times [0,1]) \cap P'_i$ non-empty?* Not in general (Example 2 below). With the restriction to $[0,1]$ removed, the predictor alone works exactly when some $a \in F(i)$ satisfies all the $y=0$ constraints (Lemma 2). Otherwise an anti-predictor is needed, and predictor plus anti-predictor always works (Lemma 3).
2. *How to represent $F(i)$?* As a box (Section 5).
3. *Do new neurons interfere with old sequences?* Not under condition (D): a new cohort stays silent during every old sequence (Lemma 1), which gives the no-forgetting part of the Theorem without storing any old sequence.

## 7 Examples

**Example 1 (a chain).** One visible neuron, $F(1) = \mathbb{R}$, sequence $1,1,1,0$. Neuron 1 is infeasible ($a \ge 1$ and $a \le 1-\varepsilon$). Its predictor has column $1,1,0,0$ and is itself infeasible (state $(1,1)$ occurs twice with different targets), so it gets a predictor with column $1,0,0,0$, whose targets are all $0$. Result: 3 neurons, key $(1,1,1)$, and for instance rows $a_1 = (0,1,0)$, $a_2 = (0,0,1)$, $a_3 = (0,0,0)$.

**Example 2 (an anti-predictor).** One visible neuron. Round 1 trains $1,1$, giving $F(1) = [1,\infty)$. Round 2 trains $1,0$, which needs $a \le 1-\varepsilon$: infeasible. The predictor has column $0,0$ and cannot help. The anti-predictor has column $1,0$ and works with a negative weight. With $\varepsilon = 1/2$ one valid result is $F(1) = [1,2]\times(-\infty,-1.5]$, $F(2) = (-\infty,0]\times(-\infty,0]$. Keys: $(1,0)$ replays $1,1$ and $(1,1)$ replays $1,0$.

## 8 Limitations and open questions

1. **Worst case is memorisation.** A full chain is a delay line, and the key then contains the sequence itself. The algorithm guarantees recall, not generalisation.
2. **Keys live outside the network.** Each trained sequence needs its key to be replayed. That is one vector per sequence, kept by the caller, playing the role $v(0)$ plays in the original notes. Apart from $v(0)$, only the part of a key belonging to its own cohort is non-zero (section 4.5).
3. **Neuron count can still grow.** The space bound is polynomial in $n$, but $n$ grows by up to $2k(n_0-|I|)$ per round when new sequences do not fit.
4. **Earlier cohorts are never reused.** New sequences are specified on visible neurons, with earlier hidden neurons at $0$. Consequently, training a sequence a second time creates a new cohort if it needed helpers the first time. A caller that keeps keys can avoid this by first running the network from each stored key whose visible part matches and skipping the sequence if it already replays.
5. **No bias term.** From the all-zero state nothing can fire. The algorithm copes by adding a predictor that is active in the key.
6. **Floating point.** $\sigma$ has a hard threshold at $1$. The code keeps the constraints as stated and instead fires when $x \ge 1 - \delta$, with $\delta = 10^{-7}$ (section 3.3). Requiring $\ge 1+\delta$ in the constraints with $y=1$ would also work.

7. **Sleep needs stored data.** The sleep phase rebuilds from a buffer of raw sequences, so unlike the wake phase it does not run on the boxes alone. Sequences that leave the buffer are forgotten at the next sleep.
8. **Sleep compresses only into finite machines.** Its machine-finding step (section 4.7) needs exact data and few input patterns, and was designed by Claude. Letters for screens, and helpers for images, are found by separate searches.
9. **Hidden neurons cannot read the inputs.** Under (D), no visible neuron can excite a hidden one, so the wake phase can only memorise; everything that generalises comes from fixed neurons installed by sleep or by hand.

Open questions: what should sleep compute in general, and can it work from the boxes and keys alone, without a buffer? Can (D) be relaxed so that hidden neurons are driven by the inputs, without losing the no-forgetting theorem? Can earlier cohorts be reused safely by later sequences? Are there helper columns other than shifts that lead to shorter chains or to generalisation? Can idle hidden neurons be merged or pruned without breaking the Theorem?

## 9 Verification

The algorithms above were implemented and tested: 300 random networks (1–4 visible neurons, with and without an input neuron, 1–4 rounds, sequences of 2–6 activation vectors, $\varepsilon \in \{0.1, 0.5, 1\}$, including new sequences that start like an old one and then diverge). After every round, all sequences trained so far were replayed with 20 weight matrices per sequence drawn from the boxes, including both extreme corners. All replays were exact. The bound of Lemma 4 and condition (D) were also checked on 200 further random networks.

**Smaller margin (26 November).** 150 random networks (1–4 visible neurons, with and without an input neuron, 1–4 rounds of 1–3 sequences of 2–6 vectors) were trained at $\varepsilon = 1/2,\ 10^{-3},\ 10^{-4},\ 10^{-6}$ and, after every round, every sequence so far was replayed with 10 weight matrices (both corners and random points). All replays were exact at every $\varepsilon$. Hidden neurons added in total: 1135 at $1/2$, 1062 at each smaller value.

**Sleep.** `dnn_sleep` and its prune rule were tested in `tests/` and used in experiments 15 and 18 (`experiments/RESULTS.md`, sections 15 and 18): recall of every buffered stream from its key was exact in every run.
