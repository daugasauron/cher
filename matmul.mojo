from math import sqrt, cos, sin, log, pi, exp
from gpu.random import Random
from layout.layout_tensor import Layout, LayoutTensor
from gpu import block_dim, block_idx, thread_idx


fn matmul[a_layout: Layout, b_layout: Layout, c_layout: Layout](
    a: LayoutTensor[DType.float32, a_layout, MutableAnyOrigin],
    b: LayoutTensor[DType.float32, b_layout, MutableAnyOrigin],
    c: LayoutTensor[DType.float32, c_layout, MutableAnyOrigin],
):
    M = a.dim(0)
    N = b.dim(1)
    K = b.dim(0)

    row = block_dim.x * block_idx.x + thread_idx.x
    col = block_dim.y * block_idx.y + thread_idx.y

    dst_reg: c.element_type = 0

    if row < M and col < N:
        for k_index in range(K):
            dst_reg = dst_reg + a[row, k_index] * b[k_index, col]

        c[row, col] = dst_reg


fn vector_add[layout: Layout](
    a: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
    b: LayoutTensor[DType.float32, layout, MutableAnyOrigin],
):

    var N = a.dim(0)
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= N:
        return

    a[tid, 0] = a[tid, 0] + b[tid, 0]


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

fn relu[layout: Layout](tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin]):
    var N = tensor.dim(0)
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= N:
        return

    tensor[tid, 0] = max(tensor[tid, 0], 0)


fn tanh[layout: Layout](tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin]):
    var N = tensor.dim(0)
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= N:
        return

    e1 = exp( tensor[tid, 0])
    e2 = exp(-tensor[tid, 0])
    tensor[tid, 0] = (e1 - e2) / (e1 + e2)

fn gbm_paths[layout: Layout](
        tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin], 
        mu: Float32, 
        sigma: Float32, 
        dt: Float32,
        seed: Int,
):
    alias simd_width = 4

    alias M = Int(layout.shape[0])
    alias N = Int(layout.shape[1])
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= M:
        return

    var thread_seed = seed + tid
    var random = Random(seed=thread_seed)
    var uni = random.step_uniform()

    var norm : SIMD[DType.float32, simd_width] = SIMD[DType.float32, simd_width]()
    norm[0] = sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1])
    norm[1] = sqrt(-2 * 2 * log(uni[0])) * sin(2 * pi * uni[1])
    norm[2] = sqrt(-2 * 2 * log(uni[2])) * cos(2 * pi * uni[3])
    norm[3] = sqrt(-2 * 2 * log(uni[2])) * sin(2 * pi * uni[3])

    tensor[tid, 0] = 1

    @parameter
    for i in range(1, N):
        k = i % simd_width

        if k == 0:
            uni = random.step_uniform()
            norm[0] = sqrt(-2 * 2 * log(uni[0])) * cos(2 * pi * uni[1])
            norm[1] = sqrt(-2 * 2 * log(uni[0])) * sin(2 * pi * uni[1])
            norm[2] = sqrt(-2 * 2 * log(uni[2])) * cos(2 * pi * uni[3])
            norm[3] = sqrt(-2 * 2 * log(uni[2])) * sin(2 * pi * uni[3])

        tensor[tid, i] = tensor[tid, i-1] * exp((mu - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * norm[k])

