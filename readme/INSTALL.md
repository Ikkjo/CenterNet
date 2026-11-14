# Installation

This port targets CPython 3.15.2 and PyTorch 2.15.0 with torchvision 0.29.2.
`uv.lock` pins all dependencies, including NumPy 2.5.4 and Numba 0.68.1.
Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python
and the project environment. System Python does not need any changes.

## Set up

From the repository root on Linux x86_64:

```sh
uv python install
uv sync --locked
uv run --directory src/lib/external python setup.py build_ext --inplace
```

`uv sync` installs the CUDA-enabled PyTorch wheel and its CUDA 13.2.2 runtime
libraries. A separate CUDA toolkit or `nvcc` is not needed for CenterNet.
The bundled legacy DCNv2 FFI extension has been replaced with torchvision's
maintained deformable convolution. Existing CenterNet convolution parameter names
and tensor shapes are preserved. The unused vendor ROI pooling examples were
removed.

For NVIDIA GPUs, install a working driver compatible with CUDA 13.x, at least
580. The RTX 3050 is supported. Check the driver and then PyTorch:

```sh
nvidia-smi
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

With Secure Boot, use signed kernel modules matching the **running kernel**.
Modules for `generic` do not serve a `lowlatency` kernel. `nvidia-smi` reports
the driver's CUDA compatibility level, not the installed toolkit version.
See [NVIDIA's compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

CPU execution is available with `--gpus -1`, including deformable convolution.
The locked environment still downloads CUDA runtime packages.

## Models and datasets

Download models from the [model zoo](MODEL_ZOO.md) into `models/`, and follow
[dataset setup](DATA.md). Run existing commands from `src` through uv, for example:

```sh
uv run --directory src python demo.py ctdet --demo ../images/16004479832_a748d55f21_k.jpg --load_model ../models/ctdet_coco_dla_1x.pth
```

Use the filename of an image present in `images/`. Demos open a visualization
window. For headless validation use the checks below.
Checkpoint inference and resumed training do not download ImageNet initialization
weights. Fresh training still uses the original backbone weight URLs.

For the optional Pascal VOC evaluator, build its CPU extensions:

```sh
uv run --directory src/tools/voc_eval_lib python setup.py build_ext --inplace
```

VOC NMS uses torchvision on CUDA when requested and available, and the CPU
extension otherwise. The old hardcoded sm_61 CUDA build is no longer required. Do not patch PyTorch's batch normalization source as the old installation
guide suggested.

## Verify the port

```sh
./scripts/check_port.sh -q
```

The checks cover DCN numerics and gradients on CPU and CUDA when available,
ResNet/ResNet-DCN/DLA/DLAv0 loss/backward/optimizer/checkpoint resume, hourglass
forward, all four detector tasks on CPU and CUDA, native NMS, Numba, and a
synthetic COCO data-loader worker plus a DLA training iteration. CPU CI runs the
same suite with CUDA cases omitted. Tests use small generated inputs; they do
not establish full COCO/KITTI accuracy or reproduce paper training results.

## Optional standalone CUDA compiler

Keep the newest toolkit separate from PyTorch's matched runtime. For CUDA 13.5:

```sh
uv venv --python 3.15.2 ~/.local/share/uv/venvs/cuda-13.5
uv pip install --python ~/.local/share/uv/venvs/cuda-13.5/bin/python 'cuda-toolkit[nvcc,cccl]==13.5.0'
export CUDA_HOME="$HOME/.local/share/uv/venvs/cuda-13.5/lib/python3.15/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

This installs compiler components in your home directory. It does not replace
system Python, drivers, or the project's CUDA runtime. For custom PyTorch CUDA
extensions use a toolkit matching PyTorch's CUDA build instead.

After downloading `ctdet_coco_dla_1x.pth`, validate the original checkpoint on a
repository image without opening a window:

```sh
uv run python scripts/check_pretrained.py models/ctdet_coco_dla_1x.pth
```

On the RTX 3050, this produced six detections above 0.3 confidence on both CPU
and CUDA, with matching top five detections. This check expects that specific
published checkpoint and image, rather than arbitrary model architectures.
