"""Compare a published DLA detection checkpoint on CPU and CUDA, without a GUI.

Run from the repo root:
uv run python scripts/check_pretrained.py models/ctdet_coco_dla_1x.pth
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/lib')]

import cv2
import numpy as np
import torch
from opts import opts
from detectors.ctdet import CtdetDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    torch.set_num_threads(2)
    image = cv2.imread(str(ROOT / 'images/16004479832_a748d55f21_k.jpg'))
    outputs = []
    devices = ['-1', '0'] if torch.cuda.is_available() else ['-1']
    for gpu in devices:
        opt = opts().init(['ctdet', '--gpus', gpu, '--load_model', args.checkpoint,
                           '--input_res', '512'])
        detector = CtdetDetector(opt)
        result = detector.run(image)
        boxes = np.concatenate([
            np.column_stack((np.full((len(b), 1), k), b))
            for k, b in result['results'].items()
        ])
        confident = boxes[boxes[:, 5] > .3]
        print('device', opt.device, 'confident detections', len(confident),
              'max score', boxes[:, 5].max())
        assert len(confident) > 0 and np.isfinite(boxes).all()
        outputs.append(boxes)
        del detector
    if len(outputs) == 2:
        cpu, cuda = [v[np.argsort(-v[:, 5])][:5] for v in outputs]
        np.testing.assert_allclose(cpu, cuda, rtol=.002, atol=.05)
        print('CPU/CUDA top five pretrained detections agree')


if __name__ == '__main__':
    main()
