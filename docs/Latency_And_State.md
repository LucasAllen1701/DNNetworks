# Latency and state

Two results about what a network can compute **without a key**, that is, on
data it was never trained on. The first is a limit that no hidden neuron can
get around. The second shows that the limit is the only one: with the right
fixed neurons every finite-state task is computed, given one extra step.

Together they sort the stream tasks of `experiments/RESULTS.md` into three
groups and explain why six of them never moved in experiments 4 and 7–9.

Lemma, Theorem and section numbers without a prefix refer to
`docs/Discrete_Neural_Network_Notes_Corrected.md` (the *notes*). Propositions
1–3 and Definitions F1–F3 are in `docs/Fixed_Neurons.md`.

## 1 Setting

**Definition L1 (machine and task).** A *machine* has a finite set of states
$Q$ with a start state $q_0$, reads a pattern $a \in \{0,1\}^k$ on $k$ input
neurons at every step, and has a next-state function $\delta(q, a)$ and an
output function $\mathrm{out}(q, a) \in \{0, 1\}$. On a stream
$x(0), x(1), \dots$ its states are $q(0) = q_0$ and
$q(t+1) = \delta(q(t), x(t))$. Only states that some stream reaches count.

The *task at latency 0* asks one output neuron $y$ for

$$y(t+1) = \mathrm{out}(q(t), x(t)), \qquad y(0) = y_0 .$$

This is the timing of every stream task in the experiments: the output at the
next step must already depend on the input of the current step. The *task at
latency 1* asks for the same outputs one step later:
$y(t+2) = \mathrm{out}(q(t), x(t))$ and $y(1) = y(0) = y_0$.

**Definition L2 (computing without a key).** A network computes the task
without a key if, for every stream, the run that starts with $y(0) = y_0$, the
first input clamped, and every other neuron at 0 produces the task's outputs
at every step.

## 2 The limit at latency 0

**Proposition 4.** Suppose some network, with any hidden or fixed neurons and
any weights, computes the task at latency 0 without a key. Then there are
weights $w \in \mathbb{R}^k$ on the inputs and one number $c_q$ per state with

$$w \cdot a + c_q \ge 1 \ \text{ if } \mathrm{out}(q, a) = 1, \qquad w \cdot a + c_q < 1 \ \text{ if } \mathrm{out}(q, a) = 0, \tag{$*$}$$

for every state $q$ and every pattern $a$.

*Proof.* By RUN, $y(t+1) = \sigma\bigl(\sum_{i \in I} A_{yi}\,x(t)_i + \sum_{j \notin I} A_{yj}\,v(t)_j\bigr)$.
The values $v(t)_j$ of the non-input neurons at time $t$ are determined by the
starting state and by the inputs $x(0), \dots, x(t-1)$; the starting state of
the non-input neurons is the same for every stream (Definition L2). So the
second sum is a number $c(h)$ that depends only on the history
$h = x(0..t-1)$, and not on $x(t)$. The first sum is $w \cdot x(t)$ with
$w = A_{y,I}$, the same for every history. For each state $q$ choose one
history $h_q$ that leads to it and put $c_q = c(h_q)$. Since the network is
right for every continuation $x(t) = a$,
$\sigma(w \cdot a + c_q) = \mathrm{out}(q, a)$, which is $(*)$. $\square$

Because there are finitely many pairs $(q, a)$, condition $(*)$ is equivalent
to the same condition with any margin $\varepsilon \le 1$ on the second
inequality: replace $w$ by $\lambda w$ and $c_q$ by $\lambda (c_q - 1) + 1$
for a large enough $\lambda$. `dnn_fixed.realisable_without_latency` decides
it with a linear program.

**Corollary (one input).** If in some state the output must copy the current
input ($\mathrm{out}(q, 0) = 0$, $\mathrm{out}(q, 1) = 1$) and in another state
it must invert it ($\mathrm{out}(q', 0) = 1$, $\mathrm{out}(q', 1) = 0$), no
network computes the task at latency 0 without a key. The first state forces
$w > 0$ and the second $w < 0$.

Parity is the simplest example: the output must copy the current input when
the parity so far is even and invert it when it is odd. The weight from the
input to the output cannot be positive and negative at once, and nothing a
hidden neuron does can change that, because every hidden neuron's value at
time $t$ was fixed before $x(t)$ arrived.

| group | tasks |
|---|---|
| $(*)$ holds | prev_input, delay2, and2, maj3, no000, Tomita 1, 2, 4 and 7 |
| $(*)$ fails: impossible at latency 0 | parity, the adder, xor2, Tomita 3, 5 and 6 |

**Remark (a key is not affected).** With a key the chain neurons carry the
answers, so the trained streams still replay. The limit concerns unseen
streams only.

**Remark (networks without inputs).** In a run without a key every hidden and
fixed neuron is 0 at time 0, so the visible state at time 1 is a threshold
function of the visible state at time 0. Rule 90 with one automaton step per
network step therefore fails at its first step whatever neurons are added.
Experiment 6 gives it two network steps per automaton step for this reason.

## 3 State neurons

**Definition L3 (state neurons).** Given a machine and the always-on input
`on`, add these fixed neurons:

- `started`, with weight 1 from `on`. It is 0 at time 0 and 1 afterwards.
- $n_{q,a}$ for every state $q$ and pattern $a$, with weights
  - $+1$ from every $n_{p,b}$ with $\delta(p, b) = q$,
  - $+1$ from input $i$ if $a_i = 1$ and $-1$ if $a_i = 0$,
  - $[q = q_0] - |a|$ from `on`, where $|a|$ is the number of 1s in $a$,
  - $-[q = q_0]$ from `started`.

All weights are integers, so these are fixed neurons (Definition F1). There
are $|Q|\,2^k + 1$ of them. This is Minsky's construction of a finite
automaton from threshold units, with the clock neuron added so that the start
state needs no key.

**Lemma S.** On every extended sequence, and on every run without a key in
which `on` is always on and the fixed neurons start at 0,

$$n_{q,a}(t+1) = 1 \iff q(t) = q \text{ and } x(t) = a \qquad (t \ge 0).$$

In particular exactly one state neuron is on at every time $t \ge 1$.

*Proof.* By Proposition 2 the state neurons follow their rule on such a run,
as they do on an extended sequence by definition, so it is enough to evaluate
the rule. The inputs contribute $m - e - |a|$ together with the $-|a|$ from
`on`, where $m$ is the number of positions with $a_i = x(t)_i = 1$ and $e$ the
number with $a_i = 0$, $x(t)_i = 1$. This is 0 if $x(t) = a$ and at most $-1$
otherwise. At $t = 0$ no state neuron is on, and `on` and `started` add
$[q = q_0]$; the sum is 1 exactly when $q = q_0 = q(0)$ and $x(0) = a$. At
$t \ge 1$, `on` and `started` cancel, and by induction the only state neuron
that is on is $n_{p,b}$ with $p = q(t-1)$, $b = x(t-1)$; it contributes 1
exactly when $\delta(p, b) = q$, that is when $q(t) = q$. $\square$

**Proposition 5.** Install the state neurons of the task's machine before
round 1. Then, for the output neuron $y$:

1. **Latency 1.** $y$'s next value is a threshold function of the visible
   state, at every margin $\varepsilon \le 1$. So $P_y$ is non-empty for every
   set of training streams and the first round adds no hidden neuron.
2. **Latency 0.** The same holds if and only if $(*)$ holds, provided the
   training streams contain every reachable combination of state and inputs
   at two consecutive steps.

*Proof.* (1) Give $y$ the weight 1 from every $n_{q,a}$ with
$\mathrm{out}(q, a) = 1$, $y_0$ from `on`, $-y_0$ from `started`, and 0 from
everything else. At $t = 0$ the sum is $y_0 = y(1)$. At $t \ge 1$, by Lemma S
the sum is $\mathrm{out}(q(t-1), x(t-1)) = y(t+1)$. The sums are integers, so
every margin is met.

(2) If $(*)$ holds, take it with margin $\varepsilon$ and give $y$ the weights
$w$ from the inputs, $c_{\delta(p,b)}$ from $n_{p,b}$, $c_{q_0}$ from `on` and
$-c_{q_0}$ from `started`. By Lemma S the sum at time $t$ is
$w \cdot x(t) + c_{q(t)}$, which is on the correct side of the threshold.
Conversely, a weight vector for $y$ gives, at each time $t \ge 1$, a sum
$w \cdot x(t) + c$ where $c$ depends only on which state neuron is on (the
output $y(t)$ is determined by it too). If every reachable combination occurs
in the training streams, these numbers satisfy $(*)$ as in the proof of
Proposition 4. $\square$

**Corollary (which tasks can be computed).** A task given by a finite machine
can be computed without a key

- at latency 1: always;
- at latency 0: exactly when $(*)$ holds.

*Proof.* Proposition 4 gives "only when". For the rest, the weight vectors in
the proof of Proposition 5 together with the state neurons form a network
that computes the task, by Lemma S and Proposition 2. $\square$

## 4 What this says about the experiments

- **Six tasks were being asked for something impossible.** Parity, the adder,
  xor2 and Tomita 3, 5 and 6 cannot be computed at the experiments' timing by
  any network, whatever hidden neurons it has. That is why shifts, the clock,
  twins and the variant's features all left them unchanged. They are fair
  tasks at latency 1.
- **Tomita 7 was the one unsolved task that is possible.** It needs state: a
  record of how many times the input has changed. State neurons solve it at
  latency 0 (experiment 10).
- **Past shifts are state neurons for a special kind of machine**, one whose
  state is the last $d$ inputs. This is why they were enough for the delayed
  input, the window tasks and Tomita 4.
- **State is the remaining problem.** Proposition 5 builds the state neurons
  from the task's definition. Nothing in TRAIN finds them: a helper chain
  stores the outputs of one stream, not the machine behind them. A library of
  generic fixed neurons finds some machines (experiment 10 reports which).
- **No forgetting is unaffected.** State neurons and library neurons are fixed
  neurons, so Proposition 1 applies to all of them.

## 5 Implementation and tests

In `dnn_fixed.py`: `machine_from_task` finds the smallest machine of a stream
task by comparing input prefixes on all continuations of a few steps;
`realisable_without_latency` decides $(*)$; `add_automaton` installs the state
neurons; `add_library` installs the variant's feature templates over a set of
neurons. `tests/test_dnn_fixed.py` checks Lemma S and Proposition 5 on random
machines and the classification above on named ones.
