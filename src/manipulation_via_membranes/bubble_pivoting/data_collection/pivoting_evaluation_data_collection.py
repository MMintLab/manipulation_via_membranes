from bubble_utils.bubble_data_collection.data_collector_base import DataCollectorBase


class PivotingEvaluationDataCollection(DataCollectorBase):
    def __init__(self, bubble_pivoting_policy, save_path, controller_method, *args, scene_name='pivoting_evaluation', **kwargs):
        self.scene_name = scene_name
        self.controller_method = controller_method
        self.bubble_pivoting_policy = bubble_pivoting_policy
        super().__init__(*args, data_path=save_path, **kwargs)


    def _get_legend_column_names(self):
        """
        Return a list containing the column names of the datalegend
        Returns:
        """
        column_names = ['SceneName', 'ControllerMethod', 'Model', 'Version', 'Tool', 'InitAngle', 'GoalAngle', 'Achieved', 'AngleDiff', 'ToolDetected', 'NSteps', 'OnlineAngleDiff']
        return column_names

    def _get_legend_lines(self, data_params):
        """
        Return a list containing the values to log inot the data legend for the data sample with file code filecode
        Args:
            data_params: <dict> containg parameters of the collected data
        Returns:
        """
        column_names = self._get_legend_column_names()
        lines = [[data_params[cn] for cn in column_names]]
        return lines

    def _collect_data_sample(self, params=None):
        """
        Collect and save data to the designed path in self.data_path
        Args:
            params:
        Returns: <dict> containing the parameters of the collected sample
        """


        # pack the score and other significant data (num_steps, ...
        data_params = self.bubble_pivoting_policy.collect_data(self.controller_method, self.scene_name)

        return data_params
