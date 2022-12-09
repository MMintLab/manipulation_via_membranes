import abc
from collections.abc import Iterable


class BubbleControllerBase(abc.ABC):

    def control(self, state_sample):
        action = self._query_controller(state_sample)
        return action

    @abc.abstractmethod
    def _query_controller(self, state_sample):
        pass

    @property
    def name(self):
        name = '{}'.format(self.__class__.__name__)
        return name


class BubbleModelController(BubbleControllerBase):

    def __init__(self, model, env, object_pose_estimator, cost_function, state_trs=None):
        self.model = model
        self.env = env
        self.object_pose_estimator = object_pose_estimator
        self.cost_function = cost_function
        self.state_trs = state_trs

    @property
    def name(self):
        name = '{}_{}'.format(self.__class__.__name__, self.env.__class__.__name__)
        return name

    def _tr_state_sample(self, state_sample):
        if self.state_trs is not None:
            if isinstance(self.state_trs, Iterable):
                for tr_i in self.state_trs:
                    state_sample = tr_i(state_sample)
            else:
                raise TypeError('Provided state transformations {} NOT SUPPORTED'.format(self.state_trs))
        return state_sample

    def control(self, state_sample):
        state_sample = self._tr_state_sample(state_sample)
        action = super().control(state_sample)
        return action


