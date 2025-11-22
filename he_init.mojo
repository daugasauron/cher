from math import sqrt, cos, sin, log, pi, exp
from gpu.random import Random
from layout.layout_tensor import Layout, LayoutTensor
from gpu import block_dim, block_idx, thread_idx


fn he_init[layout: Layout](
    tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
    seed: Int,
):
    alias simd_width = 4
    var M = tensor.dim(0)
    var N = tensor.dim(1)
    var size = M * N
    var draws = (size + simd_width - 1) // simd_width

    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= draws:
        return

    var thread_seed = seed + tid
    var random = Random(seed=thread_seed)
    var uni = random.step_uniform()

    var std_dev = Float32(2.0 / N) ** 0.5

    if tid == draws - 1:
        remainder = size % simd_width
        if remainder == 1:
            tensor.ptr[tid * simd_width + 0] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1]))
        elif remainder == 2:
            tensor.ptr[tid * simd_width + 0] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1]))
            tensor.ptr[tid * simd_width + 1] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * sin(2 * pi * uni[1]))
        elif remainder == 3:
            tensor.ptr[tid * simd_width + 0] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1]))
            tensor.ptr[tid * simd_width + 1] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * sin(2 * pi * uni[1]))
            tensor.ptr[tid * simd_width + 2] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[2])) * cos(2 * pi * uni[3]))
        elif remainder == 0:
            tensor.ptr[tid * simd_width + 0] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1]))
            tensor.ptr[tid * simd_width + 1] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[0])) * sin(2 * pi * uni[1]))
            tensor.ptr[tid * simd_width + 2] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[2])) * cos(2 * pi * uni[3]))
            tensor.ptr[tid * simd_width + 3] = Scalar[DType.float32](std_dev * sqrt(-2 * 2 * log(uni[2])) * sin(2 * pi * uni[3]))
    else:
        tensor.ptr[tid * simd_width + 0] = Scalar[DType.float32](std_dev * sqrt(-2 *2 * log(uni[0])) * cos(2 * pi * uni[1]))
        tensor.ptr[tid * simd_width + 1] = Scalar[DType.float32](std_dev * sqrt(-2 *2 * log(uni[0])) * sin(2 * pi * uni[1]))
        tensor.ptr[tid * simd_width + 2] = Scalar[DType.float32](std_dev * sqrt(-2 *2 * log(uni[2])) * cos(2 * pi * uni[3]))
        tensor.ptr[tid * simd_width + 3] = Scalar[DType.float32](std_dev * sqrt(-2 *2 * log(uni[2])) * sin(2 * pi * uni[3]))


