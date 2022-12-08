import numpy as np

from bubble_control.bubble_learning.models.bubble_dynamics_model import BubbleDynamicsModel
import torch

from manipulation_via_membranes.bubble_pivoting.datasets.combine_dataset import PivotingCombinedDataset
from manipulation_via_membranes.bubble_learning.models.bubble_dynamics_model import  BubbleDynamicsModel
from manipulation_via_membranes.bubble_learning.aux.load_model import load_model_version
from manipulation_via_membranes.bubble_learning.aux.orientation_trs import QuaternionToAxis
from manipulation_via_membranes.bubble_learning.datasets.fixing_datasets.fix_object_pose_encoding_processed_data import EncodeObjectPoseAsAxisAngleTr
from manipulation_via_membranes.bubble_pivoting.aux.load_confs import load_object_models


trs = [QuaternionToAxis(), EncodeObjectPoseAsAxisAngleTr()]
path = '/home/mireiaplanaslisbona/Documents/research'
dataset = PivotingCombinedDataset(path, transformation=trs)
        
d0 = dataset[3395]
model_data_name = '/home/mireiaplanaslisbona/Documents/research'
model = load_model_version(BubbleDynamicsModel, model_data_name, 0)

import pdb; pdb.set_trace()

model_input = model.get_model_input(d0)

ground_truth = model.get_model_output(d0)

object_model_dataset = model_input[4]
object_models = load_object_models() # TODO: Consdier doing this more efficient to avoid having to load every time

object_model_file = np.asarray(object_models[d0['object_code']].points)
print(d0['object_code'])
print('Max difference', torch.max(torch.from_numpy(object_model_file) - object_model_dataset))

model_output = model(*model_input, d0['action'])


output = model()
