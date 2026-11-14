# Discrete Neural Network Notes — corrected version

Corrected version of `Neural_Network_Notes.pdf` (Lucas Allen, November 2026). This version is dated 15 November 2026. Math is written in LaTeX notation.

## 0 What changed

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

A layered network with $h$ layers needs $h$ steps to carry an input through to the output. So the number of steps $h$ between an input and the output that answers it is the depth of the computation. Nothing in RUN or TRAIN fixes $h$: the training sequences do, by the step at which each output appears.

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

$[n] = \{1,\dots,n\}$ and time is $t \in \{0,1,2,\dots\}$. Section 2 defines the networks, section 3 runs them and section 4 trains them. Section 5 is the space bound, sections 6 and 7 answer the questions of the original notes and give examples, and section 8 lists limitations.

**References for 1.4.** M. Mézard and J.-P. Nadal (1989), Learning in feedforward layered networks: the tiling algorithm, *Journal of Physics A* 22. M. Frean (1990), The upstart algorithm: a method for constructing and training feedforward neural networks, *Neural Computation* 2. S. Fahlman and C. Lebiere (1990), The cascade-correlation learning architecture, *Advances in Neural Information Processing Systems* 2.

## 2 Definitions

**Definition 1 (activation).** $\sigma(x) = 0$ if $x < 1$ and $\sigma(x) = 1$ if $x \ge 1$, applied componentwise to vectors.

**Definition 2 (box).** A box in $\mathbb{R}^n$ is a non-empty set $\prod_{j=1}^n [l_j, u_j]$ with $l_j \in \mathbb{R}\cup\{-\infty\}$, $u_j \in \mathbb{R}\cup\{+\infty\}$, $l_j \le u_j$ (an infinite endpoint means that side is unbounded). $\mathbb{R}^n$ is a box.

**Definition 3 (network).** A discrete neural network is a tuple $(n, I, O, F, c)$ where

- $n$ is the number of neurons;
- $c : [n] \to \{0,1,2,\dots\}$ is the *cohort* label: $c(i) = 0$ means neuron $i$ is *visible* (part of the original network), and $c(i) = r \ge 1$ means $i$ is a *hidden* neuron created in training round $r$;
- $I, O$ are disjoint sets of visible neurons, the input and output neurons;
- $F(i) = \prod_j [l_{ij}, u_{ij}]$ is a box in $\mathbb{R}^n$: the allowed weight vectors *into* neuron $i$.

An untrained network has $c \equiv 0$ and $F(i) = \mathbb{R}^n$ for all $i$.

**Dormancy condition (D).** For every hidden neuron $i$ and every neuron $j$ with $c(j) \ne c(i)$: $u_{ij} \le 0$. In words, a hidden neuron can only be excited by its own cohort.

**Definition 4 (activations).** $v(t) \in \{0,1\}^n$ is the activation vector at time $t$.

## 3 Running a network

Build the weight matrix $A \in \mathbb{R}^{n\times n}$ by choosing, for each $i$, any point $a_i \in F(i)$ as the $i$-th **row** of $A$. So $A_{ij}$ is the weight of the synapse from neuron $j$ to neuron $i$. (Rows of input neurons are never used.)

**ALGORITHM RUN**

INPUTS: $A$; an initial vector $v(0) \in \{0,1\}^n$; a stream of sets $S_0, S_1, \dots \subset I$.

OUTPUT: a stream of sets $T_0, T_1, \dots \subset O$.

STEPS:

1. For each $i \in I$ set $v(0)_i := 1$ if $i \in S_0$, else $0$. Emit $T_0 = \{o \in O : v(0)_o = 1\}$.
2. For $t = 0, 1, 2, \dots$:
   1. $w := \sigma(A\,v(t))$.
   2. For each $i \in I$ set $w_i := 1$ if $i \in S_{t+1}$, else $0$.
   3. $v(t+1) := w$. Emit $T_{t+1} = \{o \in O : v(t+1)_o = 1\}$.

## 4 Training a network

Training happens in rounds $r = 1, 2, \dots$. Each round takes a finite set of new sequences and returns a larger network that has learned them without forgetting earlier ones.

### 4.1 Constraints for one neuron

Suppose the current network has $m$ neurons and each training sequence is $[v(0), \dots, v(k)]$ with $v(t) \in \{0,1\}^m$. For a neuron $g$, every sequence and every $t \in \{0,\dots,k-1\}$ gives a pair

$$x = v(t) \in \{0,1\}^m, \qquad y = v(t+1)_g \in \{0,1\}.$$

Fix a margin $\varepsilon \in (0,1]$. The weight vector $a \in \mathbb{R}^m$ into $g$ must satisfy

$$a \cdot x \ge 1 \ \text{ if } y = 1, \qquad a \cdot x \le 1 - \varepsilon \ \text{ if } y = 0.$$

Call the set of such $a$ the polyhedron $P_g$. (The margin replaces the strict inequality $<1$, which a linear program cannot express. A smaller $\varepsilon$ rejects fewer problems.)

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

**ADD(column).** In round $r$, with $m$ neurons so far:

1. $m := m+1$, $c(m) := r$, and append the column to every training sequence.
2. Lift every existing box by one coordinate (the weight from the new neuron):
   - if $j$ is visible or $c(j) = r$: $\tilde F(j) := \tilde F(j) \times \mathbb{R}$;
   - if $j$ is hidden from an earlier round: $\tilde F(j) := \tilde F(j) \times (-\infty, 0]$.
3. The new neuron's box is $\tilde F(m) := \prod_{j=1}^m J_j$ with $J_j = (-\infty, 0]$ if $c(j) \ne r$ and $J_j = \mathbb{R}$ if $c(j) = r$ (including $j = m$).

Steps 2 and 3 are exactly what keeps condition (D) true.

### 4.4 The training algorithm

**ALGORITHM TRAIN (round $r$)**

INPUTS: a network $(n, I, O, F, c)$ satisfying (D); a finite set $V$ of sequences $[v(0),\dots,v(k)]$ of activation vectors **on the visible neurons** (lengths may differ; the input neurons' values are part of the sequences); a margin $\varepsilon \in (0,1]$.

OUTPUT: a network $(n', I, O, F', c')$, and for each sequence in $V$ its extended initial vector $v'(0) \in \{0,1\}^{n'}$ (its *key*).

STEPS:

0. Extend every sequence in $V$ by zeros on all hidden neurons. Set $\tilde F := F$, $m := n$. Let the worklist $W$ and the refit list $R$ both be the visible non-input neurons.
1. While $W$ is non-empty, remove a neuron $g$ from $W$.
   - If FEAS($g$) holds, continue.
   - (a) Otherwise ADD the predictor $p$ of $g$. If FEAS($g$) now holds, put $p$ in $W$ and $R$ and continue. (For a hidden $g$ this always succeeds, by Lemma 2.)
   - (b) Otherwise ($g$ is visible) undo the addition of $p$ and ADD the anti-predictor $q$ of $g$ instead. If FEAS($g$) now holds, put $q$ in $W$ and $R$ and continue.
   - (c) Otherwise undo the addition of $q$ and ADD both $p$ and $q$. FEAS($g$) now holds by Lemma 3. Put $p$ and $q$ in $W$ and $R$.
2. For each $g \in R$ set $F'(g) :=$ SUBBOX($g$). For every other neuron (input neurons and hidden neurons of earlier rounds) set $F'(g) := \tilde F(g)$.
3. Return $(m, I, O, F', c)$ together with $v'(0)$ for each sequence.

To run a learned sequence, call RUN with its key as the initial vector and its input sets $S_t$. A key from an earlier round is padded with zeros for neurons created later.

### 4.5 Correctness

**Lemma 1 (dormancy).** Assume (D). If every neuron of a cohort $C$ is $0$ at time $t$, then every neuron of $C$ is $0$ at time $t+1$, for every choice of $A$.

*Proof.* For $h \in C$ and $a \in F(h)$: $a \cdot v(t) = \sum_{j \notin C} a_j v(t)_j \le 0 < 1$, since $a_j \le 0$ for $j \notin C$ and $v(t)_j = 0$ for $j \in C$. $\square$

**Lemma 2 (predictor).** Suppose some $a \in \tilde F(g)$ satisfies every constraint of $g$ with $y = 0$. Then after adding the predictor of $g$, FEAS($g$) holds. In particular this applies whenever $0 \in \tilde F(g)$: to every neuron in round 1, and to every hidden neuron of the current round.

*Proof.* Take the new weight $b = \max(0,\, 1 - \min a\cdot x)$, the minimum over pairs with $y = 1$. On pairs with $y=1$ the predictor is $1$, so $a\cdot x + b \ge 1$. On pairs with $y = 0$ it is $0$, so the value is $a \cdot x \le 1-\varepsilon$. The new coordinate of $g$ is unrestricted, because $g$ is visible or belongs to the current cohort. If $0 \in \tilde F(g)$, then $a = 0$ works since $0 \le 1 - \varepsilon$. $\square$

**Lemma 3 (predictor and anti-predictor).** Let $g$ be visible. After adding both $p$ and $q$, FEAS($g$) holds.

*Proof.* Take any finite $a \in \tilde F(g)$ and let $L = \min a\cdot x$, $U = \max a \cdot x$ over all pairs. Give $p$ the weight $b_p = \max(0, 1 - L)$ and $q$ the weight $b_q = \min(0, 1 - \varepsilon - U)$. On pairs with $y = 1$: $p = 1$, $q = 0$, value $a\cdot x + b_p \ge 1$. On pairs with $y=0$: $p = 0$, $q = 1$, value $a \cdot x + b_q \le 1-\varepsilon$. $\square$

**Lemma 4 (termination).** Let $k$ be the length of the longest sequence in $V$. Step 1 terminates, and the round adds at most $2k\,(n_0 - |I|)$ neurons, where $n_0$ is the number of visible neurons.

*Proof.* Helpers of helpers are always predictors (Lemma 2), so each helper of a visible neuron starts a chain $h_1, h_2, \dots$ where $h_{s+1}$ is the predictor of $h_s$. Then $h_s(t) = h_1(t+s-1)$ when $t+s-1 \le k-1$ and $0$ otherwise. The targets of $h_k$ are therefore all $0$, so $a = 0$ is feasible and the chain stops with at most $k$ neurons. Each visible non-input neuron starts at most two chains. $\square$

**Theorem.** After any number of rounds, for every weight matrix $A$ whose rows lie in the boxes:

1. *(Learning)* every sequence of the latest round, run from its key with its inputs, is reproduced exactly on all neurons;
2. *(No forgetting)* every sequence of every earlier round, run from its zero-padded key, is still reproduced exactly;
3. condition (D) holds;
4. the network is stored in $2n^2 + n$ numbers, where $n$ is the current number of neurons.

*Proof.* (3) holds by the lift rule in ADD, and SUBBOX only shrinks boxes. (4) is Definition 3.

(1) Induct on $t$. Input neurons are clamped to the sequence's values. For $g \in R$, $F'(g) \subset P_g$, so $\sigma(a_g \cdot v'(t)) = v'(t+1)_g$. Hidden neurons of earlier rounds are $0$ in the extended sequence and stay $0$ by Lemma 1.

(2) Let $[v(0),\dots,v(k)]$ be a sequence learned before round $r$, and $A'$ a matrix for the new network. Induct on $t$ with the claim that the state is $(v(t), 0)$. The new cohort is $0$ at time $t$, so it is $0$ at time $t+1$ by Lemma 1. For an old neuron $i$, the row $a'_i \in F'(i) \subset F(i) \times (\text{intervals})$ restricts to some $a_i \in F(i)$, and $a'_i \cdot (v(t), 0) = a_i \cdot v(t)$. So the old neurons behave as in the old network, which reproduces the sequence for every choice of rows in $F$. $\square$

## 5 The space bound

**What is stored.** $2n$ interval endpoints per neuron plus one cohort label: $2n^2 + n$ numbers. This does not depend on the number or length of sequences trained. The training sequences of a round are needed only during that round.

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
2. **Keys live outside the network.** Each trained sequence needs its key to be replayed. That is one vector per sequence, kept by the caller, playing the role $v(0)$ plays in the original notes. Only the part of a key belonging to its own cohort is non-zero.
3. **Neuron count can still grow.** The space bound is polynomial in $n$, but $n$ grows by up to $2k(n_0-|I|)$ per round when new sequences do not fit.
4. **Earlier cohorts are never reused.** New sequences are specified on visible neurons, with earlier hidden neurons at $0$. Consequently, training a sequence a second time creates a new cohort if it needed helpers the first time. A caller that keeps keys can avoid this by first running the network from each stored key whose visible part matches and skipping the sequence if it already replays.
5. **No bias term.** From the all-zero state nothing can fire. The algorithm copes by adding a predictor that is active in the key.
6. **Floating point.** $\sigma$ has a hard threshold at $1$. In floating point, use $\ge 1+\delta$ for a small $\delta$ in the constraints with $y=1$.

Open questions: can earlier cohorts be reused safely by later sequences? Are there helper columns other than shifts that lead to shorter chains or to generalisation? Can idle hidden neurons be merged or pruned without breaking the Theorem?

## 9 Verification

The algorithms above were implemented and tested: 300 random networks (1–4 visible neurons, with and without an input neuron, 1–4 rounds, sequences of 2–6 activation vectors, $\varepsilon \in \{0.1, 0.5, 1\}$, including new sequences that start like an old one and then diverge). After every round, all sequences trained so far were replayed with 20 weight matrices per sequence drawn from the boxes, including both extreme corners. All replays were exact. The bound of Lemma 4 and condition (D) were also checked on 200 further random networks.
