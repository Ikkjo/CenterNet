from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


# Importable classes let spawn/forkserver workers unpickle dataset instances.
for dataset_name, dataset_class in dataset_factory.items():
  for task_name, sample_class in _sample_factory.items():
    name = '{}_{}_Dataset'.format(dataset_name, task_name)
    globals()[name] = type(name, (dataset_class, sample_class),
                           {'__module__': __name__})


def get_dataset(dataset, task):
  # Preserve the factory's KeyError for unknown datasets and tasks.
  dataset_factory[dataset]
  _sample_factory[task]
  return globals()['{}_{}_Dataset'.format(dataset, task)]
