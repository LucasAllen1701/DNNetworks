import numpy as np

import dnn_fixed as F
import dnn_sleep as S
from tests.test_dnn import replays


def _layout_with_silent_neuron():
    lay = F.Layout(2, [0], [1])          # input 0, output 1 = previous input
    lay.add_on()
    lay.add_fixed({0: 1}, "copy")        # copy of the input
    lay.add_fixed({lay.on: -1}, "silent")  # always 0
    return lay


def test_prune_deletes_a_silent_neuron_and_keeps_recall():
    rng = np.random.default_rng(0)
    lay = _layout_with_silent_neuron()
    net = F.make_net(lay)
    V = []
    for _ in range(4):
        x = rng.integers(0, 2, 9)
        V.append(np.c_[x, np.r_[0, x[:-1]]])
    trained = F.train(net, lay, V)
    assert replays(net, trained, rng)
    net2, lay2, trained2, df, dh = S.prune(net, lay, trained)
    assert df >= 1 and "silent" not in lay2.names
    assert replays(net2, trained2, rng)
    ext = [F.extend(lay2, s) for s in V]
    assert all((e[:, :2] == t[:, :2]).all() for e, t in zip(ext, trained2))


def test_prune_never_deletes_task_or_on_neurons():
    lay = _layout_with_silent_neuron()
    net = F.make_net(lay)
    net2, lay2, _, _, _ = S.prune(net, lay, [])
    assert lay2.names[:3] == ["v0", "v1", "on"]


def test_witness_eviction_keeps_the_machine():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import exp13_state_merging as e13
    import dnn_merge as M
    out, V, _ = e13.data("tomita4", 24, 0)
    kept, dropped = S.evict(V, 16, out, "witness")
    assert len(kept) == 16
    if dropped == 0:
        assert S._same(M.learn_machine(kept, out), M.learn_machine(V, out))
    assert S.evict(V, 16, out, "recent")[0] == V[-16:]
