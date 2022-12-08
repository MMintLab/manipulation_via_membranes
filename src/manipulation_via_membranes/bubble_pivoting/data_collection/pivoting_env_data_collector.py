import numpy as np
import csv
import tqdm
import time
from bubble_utils.bubble_data_collection.env_data_collection import ReferencedEnvDataCollector
from manipulation_via_membranes.bubble_pivoting.pivoting_model_control.aux.pivoting_geometry import get_angle_difference, get_tool_axis


class PivotingEnvDataCollector(ReferencedEnvDataCollector):
    
    def _collect_data_sample(self, i, params=None):
        # Get initial observation
        init_fc = self.get_new_filecode()
        final_fc = self.get_new_filecode()
        failed = True
        while failed:
            failed = False
            tool = self.env.tool_detected_listener.get(block_until_data=True).data
            if not tool:
                print('No tool detected')
                i = 0 
                failed = True
            tool_axis_gf = get_tool_axis(ref_frame='grasp_frame')
            tool_axis_wf = get_tool_axis(ref_frame='med_base')
            if tool_axis_wf[2] > 0:
                tool_axis_gf *= -1
            tool_angle_gf = get_angle_difference(child_axis=tool_axis_gf, parent_axis=np.array([0, 0, 1]))
            if np.abs(tool_angle_gf) > 5 * np.pi/6 or i == 0:
                print('Resetting')
                if i == 0:
                    print('Because i == 0')
                else:
                    print('Because np.abs(tool_angle_gf) > 5 * np.pi/6')
                self.env.no_tool_reset()
                pre_action, valid_pre_action = self.env.get_action(action_space=self.env.init_action_space, is_action_valid=self.env.is_pre_action_valid)
                if not valid_pre_action:
                    print('Pre-action: {} -- NOT VALID!'.format(pre_action))
                    print('Resetting')
                    self.env.no_tool_reset()
                    failed = True
                    continue
                init_feedback = self.env.do_pre_action_init(pre_action)
                tool = self.env.tool_detected_listener.get(block_until_data=True).data
                if not tool:
                    print('No tool detected')
                    self.env.no_tool_reset()
                    failed = True
                    continue
                if init_feedback['planning_success'] == False or init_feedback['execution_success'] == False:
                    print('Initial pose plan {} failed'.format(pre_action))
                    failed = True
                    continue
                lower_feedback = self.env.do_pre_action_lower()
                if lower_feedback['planning_success'] == False or lower_feedback['execution_success'] == False:
                    print('Lowering failed')
                    failed = True
                    continue            
                self.env._do_pre_action_prepare()
            else:
                self.env._do_pre_action_prepare(open_width=None)
            tool = self.env.tool_detected_listener.get(block_until_data=True).data
            if not tool:
                print('No tool detected')
                failed = True
                i = 0
                continue
            self.action_space = self.env._get_action_space()
            self.max_force_felt = 0
            init_obs = self.env.get_observation()
            init_obs.modify_data_params(self.data_save_params)
            init_obs.save_fc(init_fc)
            obs_columns = list(init_obs.keys())
            # Get action
            if self.manual_actions:
                action = self._get_action_from_input()
                action = self.env._tr_action_space(action)
                valid = self.env.is_valid_action(action)
            else:
                action, valid = self.env.get_action(action_space=self.action_space, is_action_valid=self.env.is_action_valid)
            if not valid:
                print('Action: {} -- NOT VALID!'.format(action))
                failed = True
                i = 0
                continue
            else:
                # get one step sample:
                observation, reward, done, info = self.env.step(action)
                if info['planning_success'] == False or info['execution_success'] == False:
                    # i = 0
                    print('Action failed')
                    failed = True
                    continue       
                observation.modify_data_params(self.data_save_params)
                observation.save_fc(final_fc)
            sample_params = {
                'init_fc': init_fc,
                'final_fc': final_fc,
                'init_obs': init_obs,
                'obs': observation,
                'action': action,
                'reward': reward,
                'done': done,
                'info': info,
                'valid': valid,
            }
        return sample_params
    
    def _collect_data_sample(self, i, params=None):
        if self.ref_obs is not self.env.ref_obs:
            # record the reference
            self._record_reference_state()
        sample_params = super()._collect_data_sample(i, params=params)
        info = sample_params['info']
        planning_success = True
        execution_success = True
        if 'planning_success' in info and 'execution_success' in info:
            planning_success = info['planning_success']
            execution_success = info['execution_success']
        if sample_params['done'] or not sample_params['valid'] or not planning_success or not execution_success:
            self.env.initialize()
        return sample_params

    def collect_data(self, num_data):
        # Display basic information
        print('_____________________________')
        print(' Data collection has started!')
        print('  - The data will be saved at {}'.format(self.data_path))

        # Collect data
        pbar = tqdm(range(num_data), desc='Data Collected: ')
        num_data_collected = 1
        self.data_stats['to_collect'] = num_data
        self.data_stats['collected'] = 0
        for i in pbar:
            pbar.set_postfix({'Filecode': self.filecode})
            self.data_stats['collected'] = i
            # Save data
            sample_params = self._collect_data_sample(i)
            # Log data sample info to data legend
            legend_lines_vals = self._get_legend_lines(sample_params)
            num_data_collected = len(legend_lines_vals)
            with open(self.datalegend_path, 'a+') as csv_file:
                csv_file_writer = csv.writer(csv_file)
                for line_val in legend_lines_vals:
                    csv_file_writer.writerow(line_val)
            csv_file.close() # make sure it is closed
            # Update the filecode
            self._save_filecode_pickle()
            time.sleep(0.5)
            self.data_stats['collected'] += 1