"""CPU-importable contract; import the GPU implementation only at launch time."""
import math
from bench.api import FIELDS

BLOCKS = (128, 256, 512)


def supported(shape):
    if any(not isinstance(getattr(shape, key), int) or isinstance(getattr(shape, key), bool)
           for key in FIELDS):
        return False
    if min(shape.n, shape.cin, shape.cout, shape.h, shape.w, shape.groups) <= 0:
        return False
    return (shape.cin == shape.cout and shape.cin % shape.groups == 0
            and shape.cin // shape.groups in (1, 2, 4)
            and shape.r == shape.s == 3
            and shape.sh == shape.sw == shape.dh == shape.dw == 1
            and shape.ph == shape.pw == 1
            and max(math.prod(shape.input), math.prod(shape.weight)) < 2**31)


def make_call(shape, x, weights, output, block):
    if not supported(shape):
        raise ValueError('UNSUPPORTED: Triton requires B geometry and cpg 1/2/4')
    if type(block) is not int or block not in BLOCKS:
        raise ValueError('block must be 128, 256 or 512')
    import torch
    for value, dims in ((x, shape.input), (weights, shape.weight), (output, shape.output)):
        if (value.dtype != torch.float32 or value.device.type != 'cuda'
                or tuple(value.shape) != tuple(dims) or not value.is_contiguous()):
            raise ValueError('expected contiguous CUDA FP32 NCHW tensors')
    if x.device != weights.device or x.device != output.device:
        raise ValueError('device mismatch')
    # Reject overlapping storage even when a tensor uses a nonzero storage offset.
    intervals = [(v.data_ptr(), v.data_ptr() + v.numel() * v.element_size())
                 for v in (x, weights, output)]
    if any(max(a[0], b[0]) < min(a[1], b[1])
           for i, a in enumerate(intervals) for b in intervals[i + 1:]):
        raise ValueError('tensor buffers must not overlap')
    from .kernel import direct
    import triton
    def call():
        call.compiled = direct[(triton.cdiv(math.prod(shape.output), block),)](
            x, weights, output, shape.n, shape.cin, shape.h, shape.w,
            shape.cin // shape.groups, BLOCK=block, num_warps=4,
            num_stages=1, enable_fp_fusion=False)
        return output
    return call
