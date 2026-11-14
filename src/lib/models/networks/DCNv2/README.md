# Deformable convolution

CenterNet's `DCN` and `DCNv2` layers use `torchvision.ops.deform_conv2d`.
Install matching torch and torchvision wheels with `uv sync`; no CUDA compiler
or separate extension build is required. CPU and CUDA forward/backward are supported.

The weight, bias, and `conv_offset_mask` parameter names and shapes retain the
original CenterNet checkpoint layout. Offsets remain interleaved y/x, and masks
use sigmoid modulation. A zero offset and zero mask logit produce half of the
ordinary convolution plus the unscaled bias.

The unused vendor deformable ROI pooling examples and PyTorch 0.4 FFI build
sources were removed. CenterNet does not use deformable ROI pooling.
The original vendor license remains in LICENSE.
