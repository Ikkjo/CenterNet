"""Keep the standalone evaluator's generic module names out of the main imports."""
import subprocess
import sys
from pathlib import Path


def test_voc_native_extensions():
    subprocess.run([sys.executable, '-c', '''
import numpy as np
from model.nms_wrapper import nms
from utils.cython_bbox import bbox_overlaps
boxes = np.array([[0,0,10,10,.9],[0,0,10,10,.8],[30,30,40,40,.7]], dtype=np.float32)
assert nms(boxes, .5, force_cpu=True) == [0,2]
assert nms(boxes, .5) == [0,2]
coords = np.ascontiguousarray(boxes[:,:4], dtype=np.float64)
np.testing.assert_allclose(bbox_overlaps(coords, coords).diagonal(), 1)
'''], cwd=Path(__file__).resolve().parents[1] / 'src/tools/voc_eval_lib', check=True)
