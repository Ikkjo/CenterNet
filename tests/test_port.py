"""Runtime checks for the maintained DCN replacement and current autograd."""
import pytest
import torch
from torch.nn.functional import conv2d
from models.networks.DCNv2.dcn_v2 import DCN
from models.utils import _sigmoid

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_dcn_initialization_and_backward(device):
    torch.manual_seed(0)
    layer = DCN(3, 4, 3, 1, 1).to(device)
    x = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
    actual = layer(x)
    expected = conv2d(x, layer.weight, None, padding=1) * .5
    expected = expected + layer.bias[None, :, None, None]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for value in [x, *layer.parameters()]:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    assert layer.conv_offset_mask.weight.grad.abs().sum() > 0
    clone = DCN(3, 4, 3, 1, 1).to(device)
    clone.load_state_dict(layer.state_dict(), strict=True)
    torch.testing.assert_close(clone(x), actual)


def test_heatmap_backward():
    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    _sigmoid(logits).log().sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('arch', ['res_18', 'resdcn_18', 'dla_34', 'dlav0_34'])
def test_model_training_and_checkpoint(arch, tmp_path):
    from models.model import create_model, save_model, load_model
    from trains.ctdet import CtdetLoss
    from opts import opts
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    opt = opts().init(['ctdet', '--arch', arch, '--input_res', '64'])
    model = create_model(arch, opt.heads, 16, pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = {
        'hm': torch.zeros(2, 80, 16, 16, device=device),
        'reg_mask': torch.ones(2, 1, dtype=torch.uint8, device=device),
        'ind': torch.zeros(2, 1, dtype=torch.long, device=device),
        'wh': torch.ones(2, 1, 2, device=device),
        'reg': torch.zeros(2, 1, 2, device=device),
    }
    batch['hm'][:, 0, 0, 0] = 1
    outputs = model(torch.randn(2, 3, 64, 64, device=device))
    loss, _ = CtdetLoss(opt)(outputs, batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    path = tmp_path / 'checkpoint.pth'
    save_model(path, 3, model, optimizer)
    restored = create_model(arch, opt.heads, 16, pretrained=False).to(device)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    _, _, epoch = load_model(restored, path, restored_optimizer, True, 1e-4, [90])
    assert epoch == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('task', ['ctdet', 'multi_pose', 'ddd', 'exdet'])
def test_detector_pipeline(device, task, tmp_path):
    import numpy as np
    from opts import opts
    from models.model import create_model, save_model
    from detectors.detector_factory import detector_factory
    opt = opts().init([task, '--arch', 'res_18', '--input_res', '64',
                       '--head_conv', '16', '--K', '6', '--gpus',
                       '0' if device == 'cuda' else '-1'])
    path = tmp_path / 'detector.pth'
    model = create_model(opt.arch, opt.heads, opt.head_conv, pretrained=False)
    save_model(path, 0, model)
    opt.load_model = str(path)
    detector = detector_factory[task](opt)
    result = detector.run(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result['results']
    for boxes in result['results'].values():
        assert np.isfinite(np.asarray(boxes)).all()


def test_hourglass_forward():
    from models.model import create_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model('hourglass', {'hm': 3, 'wh': 2, 'reg': 2}, 16,
                         pretrained=False).to(device).eval()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 128, 128, device=device))
    assert len(outputs) == 2
    for output in outputs:
        assert output['hm'].shape == (1, 3, 32, 32)
        assert all(torch.isfinite(v).all() for v in output.values())


def test_coco_loader_and_trainer(tmp_path):
    import json
    import cv2
    import numpy as np
    from datasets.dataset_factory import get_dataset
    from models.model import create_model
    from opts import opts
    from trains.ctdet import CtdetTrainer
    folder = tmp_path / 'coco'
    (folder / 'annotations').mkdir(parents=True)
    (folder / 'train2017').mkdir()
    cv2.imwrite(str(folder / 'train2017' / 'sample.jpg'),
                np.full((64, 64, 3), 128, dtype=np.uint8))
    data = {'images': [{'id': 1, 'file_name': 'sample.jpg', 'width': 64, 'height': 64}],
            'categories': [{'id': 1, 'name': 'person'}],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                             'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}]}
    (folder / 'annotations' / 'instances_train2017.json').write_text(json.dumps(data))
    opt = opts().init(['ctdet', '--arch', 'dla_34', '--input_res', '64',
                       '--batch_size', '2', '--num_iters', '1', '--print_iter', '1'])
    opt.data_dir = str(tmp_path)
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset('coco', 'ctdet')(opt, 'train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1,
                                        sampler=[0, 0])
    model = create_model('dla_34', opt.heads, 16, pretrained=False)
    trainer = CtdetTrainer(opt, model, torch.optim.Adam(model.parameters()))
    trainer.set_device([0] if opt.device.type == 'cuda' else [-1], [2], opt.device)
    stats, _ = trainer.train(1, loader)
    assert np.isfinite(stats['loss']) and stats['loss'] > 0


def test_native_nms_and_numba_oracle():
    import numpy as np
    from external.nms import nms, soft_nms
    from utils.oracle_utils import gen_oracle_map
    boxes = np.array([[0, 0, 10, 10, .9], [0, 0, 10, 10, .8],
                      [30, 30, 40, 40, .7]], dtype=np.float32)
    assert nms(boxes.copy(), .5) == [0, 2]
    adjusted = boxes.copy()
    soft_nms(adjusted, Nt=.5, method=2)
    assert adjusted[0, 4] == boxes[0, 4]
    assert adjusted[:, 4].min() < .8
    features = np.array([[[2., 3.]]], dtype=np.float32)
    result = gen_oracle_map(features, np.array([[5]], dtype=np.int64), 4, 4)
    np.testing.assert_equal(result[0, :, 1, 1], [2., 3.])


@pytest.mark.parametrize('task', ['multi_pose', 'ddd', 'exdet'])
def test_task_training_backward(task):
    from opts import opts
    from models.model import create_model
    from trains.train_factory import train_factory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = opts().init([task, '--arch', 'res_18', '--input_res', '64'])
    opt.device = device
    model = create_model('res_18', opt.heads, 16, pretrained=False).to(device)
    trainer = train_factory[task](opt, model, torch.optim.Adam(model.parameters()))
    batch = {
        'input': torch.randn(2, 3, 64, 64, device=device),
        'reg_mask': torch.ones(2, 1, dtype=torch.uint8, device=device),
        'ind': torch.zeros(2, 1, dtype=torch.long, device=device),
    }
    for head, channels in opt.heads.items():
        if head.startswith('hm'):
            batch[head] = torch.zeros(2, channels, 16, 16, device=device)
            batch[head][:, 0, 0, 0] = 1
        else:
            batch[head] = torch.ones(2, 1, channels, device=device)
    if task == 'multi_pose':
        batch.update(hps_mask=torch.ones(2, 1, 34, device=device),
                     hp_mask=torch.ones(2, 1, device=device),
                     hp_ind=batch['ind'])
    elif task == 'ddd':
        batch.update(rot_mask=batch['reg_mask'],
                     rotbin=torch.ones(2, 1, 2, dtype=torch.long, device=device),
                     rotres=torch.zeros(2, 1, 2, device=device))
    else:
        for part in 'tlbr':
            batch['ind_' + part] = batch['ind']
    _, loss, _ = trainer.model_with_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    trainer.optimizer.step()
