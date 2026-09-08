"""Direct FP32 grouped cross-correlation, one output element per lane."""
import triton
import triton.language as tl


@triton.jit
def direct(X, Wt, Y, N: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
           W: tl.constexpr, CPG: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < N * C * H * W
    ow = index % W
    oh = index // W % H
    co = index // (H * W) % C
    batch = index // (C * H * W)
    group = co // CPG
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for ci in tl.static_range(CPG):
        for r in tl.static_range(3):
            ih = oh + r - 1
            for s in tl.static_range(3):
                iw = ow + s - 1
                xv = tl.load(X + ((batch * C + group * CPG + ci) * H + ih) * W + iw,
                             valid & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W), other=0.0)
                weight = tl.load(Wt + ((co * CPG + ci) * 3 + r) * 3 + s,
                                 valid, other=0.0)
                acc = tl.fma(xv, weight, acc)
    tl.store(Y + index, acc, valid)
