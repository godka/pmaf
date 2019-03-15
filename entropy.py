import numpy as np
def generate(act_size, act, entropy_expected):
    eps = 3e-3
    _act_prob = 1. / act_size
    _act = np.zeros((act_size))
    _act.fill(_act_prob)
    _entropy = np.sum(-_act* np.log(_act))
    _index = 0
    while _entropy - entropy_expected > 0.:
        _idx = np.random.randint(act_size - 1)
        #np.argmax(_act[1:])
        _idx += 1
        _delta = eps * np.random.random()
        if _act[_idx] - _delta < 0.:
            _delta = (_act[_idx] - 1e-10) / 2.
        _act[_idx] -= _delta
        _act[0] += _delta
        _entropy = np.sum(-_act* np.log(_act))
        _index += 1
    np.random.shuffle(_act[1:])
    _tmp = _act[0]
    _act[0] = _act[act]
    _act[act] = _tmp
    return _act, _entropy

import time
t = time.time()
_arr = 0.
for p in range(100):
    _act, _entropy = generate(6,3,1.3)
    print(_act)
    _arr += np.abs(_entropy - 1.3)
print(time.time() - t, 's', _arr / 100.)