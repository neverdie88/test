from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class JointBaseline(Baseline):
    def __init__(self, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]
        self._components = tuple(components)

    @overrides
    def get_param_values(self, **tags):
        return [self._components[i].get_param_values(**tags) for i in xrange(len(self._components))]

    @overrides#Thien: not implemented yet
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def divideComponentPath(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def fit(self, observations, rewards, returns):
        for i in xrange(len(self._components)):
            path = {}
            path["observations"] = observations[:,i]
            path["returns"] = returns[:,i]
            path["rewards"] = rewards[:, i]
            self._components[i].fit([path])

    @overrides
    def predict(self, path):
        predictedVals = []
        for i in xrange(len(self._components)):
            tempPath = {}
            tempPath["observations"] = path["observations"][:,i]#extract from the joint product
            tempPath["rewards"] = path["rewards"][:, i]
            predictedVals.append(self._components[i].predict(tempPath))
        return np.asarray(predictedVals)
