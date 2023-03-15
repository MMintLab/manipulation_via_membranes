from mmint_tools.wrapping_utils.wrapping_utils import AttributeWrapper
from bubble_drawing.aux.action_spaces import DiscreteElementSpace


class ControlledEnvWrapper(AttributeWrapper):
    """
    Adds a controller on top of an environment so the action returned by get_action will be controlled.
    """

    def __init__(self, env, controller, random_action_prob=0, controlled_action_keys=None):
        """

        Args:
            env:
            controller:
            random_action_prob: <float> between [0,1], probability of sampling a random action
            controlled_action_keys: <list of string> representing the actions to control. The rest would be random. If None, all actions will be controlled.
        """
        self.controller = controller
        self.random_action_prob = random_action_prob
        self.controlled_action_keys = controlled_action_keys
        super().__init__(env)
        self.env.action_space = self._get_action_space() # modify the action space to include the action controller
        self.observation = self.env.get_observation()

    @property
    def env(self):
        return self.wrapped_object

    @classmethod
    def get_name(cls):
        return 'controlled_env'

    def _get_action_space(self):
        action_space = self.env._get_action_space()
        action_space['controller'] = DiscreteElementSpace(['random', '{}'.format(self.controller.name)], probs=[self.random_action_prob, 1-self.random_action_prob])
        return action_space

    def get_action(self):
        # We extend the original action dictionary to record weather or not the action came from a random or controlled policy
        random_action, valid_random_action = self.env.get_action()
        action_controller = random_action['controller']
        if action_controller == 'random':
            # Random action (sample env action space)
            return random_action, valid_random_action
        else:
            # Use the controller
            self.observation = self.env.get_observation()
            controlled_action = self.controller.control(self.observation)
            # NOTE: Some controllers like MPPI have internal parameters that store previously computed information to improve sample efficiency.
            #       Here, we will not update any information when we have random actions.
            # pack the action with the right order
            valid_controlled_action = self.env.is_action_valid(controlled_action)
            controlled_action['controller'] = action_controller

            if self.controlled_action_keys is not None:
                random_action_keys = [k for k in random_action.keys() if k not in self.controlled_action_keys]
                for random_key in random_action_keys:
                    controlled_action[random_key] = random_action[random_key] # replace the controlled

            return controlled_action, valid_controlled_action
