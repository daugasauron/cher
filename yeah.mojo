from sys import has_accelerator, simd_width_of
from layout.layout_tensor import Layout, LayoutTensor
from matmul import he_init, vector_add, relu, tanh, matmul
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute


fn print_matrix[layout: Layout](ctx: DeviceContext, tensor: LayoutTensor, buffer: DeviceBuffer[DType.float32]) raises:
    M = tensor.dim(0)
    N = tensor.dim(1)

    host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    host_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](host_buffer)
    l2_weight_host_tensor = LayoutTensor[DType.float32, layout, MutableAnyOrigin](host_buffer)
    ctx.enqueue_copy(dst_buf=host_buffer, src_buf=buffer)
    ctx.synchronize()

    print()
    for i in range(M):
        for j in range(N):
            print(host_tensor[i, j], '\t', end='')
        print()


fn get_ctx() raises -> DeviceContext:
    @parameter
    if not has_accelerator():
        raise Error('No compatible GPU found')
    else:
        ctx = DeviceContext()
        print('Running on', ctx.name())
        print('MULTIPROCESSOR_COUNT                ', ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT))
        print('MAX_BLOCKS_PER_MULTIPROCESSOR       ', ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR))
        print('MAX_THREADS_PER_BLOCK               ', ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK))
        print('WARP_SIZE                           ', ctx.get_attribute(DeviceAttribute.WARP_SIZE))
        print('MAX_SHARED_MEMORY_PER_BLOCK         ', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK))
        print('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
        print()

    return ctx

fn main() raises:
    ctx = get_ctx()

    alias simd_width = simd_width_of[DType.float32]()
    alias random_size: Int = 4;

    alias input_size: Int = 3
    alias output_size: Int = 1
    alias input_layout = Layout.row_major(input_size, 1)

    alias l1_size: Int = 16;
    alias l1_weight_layout = Layout.row_major(l1_size, input_size)
    alias l1_bias_layout = Layout.row_major(l1_size, 1)
    alias l1_out_layout = Layout.row_major(l1_size, 1)
    alias l1_draws = (input_size * l1_size + random_size - 1) // random_size

    alias l2_size: Int = 16;
    alias l2_weight_layout = Layout.row_major(l2_size, l1_size)
    alias l2_bias_layout = Layout.row_major(l2_size, 1)
    alias l2_out_layout = Layout.row_major(l2_size, 1)
    alias l2_draws = (l1_size * l2_size + random_size - 1) // random_size

    alias l3_size: Int = 1;
    alias l3_weight_layout = Layout.row_major(l3_size, l2_size)
    alias l3_bias_layout = Layout.row_major(l3_size, 1)
    alias l3_out_layout = Layout.row_major(l3_size, 1)
    alias l3_draws = (l2_size * l3_size + random_size - 1) // random_size

    input_buffer = ctx.enqueue_create_buffer[DType.float32](input_size)
    input_host_buffer = ctx.enqueue_create_host_buffer[DType.float32](input_size)
    input_tensor = LayoutTensor[DType.float32, input_layout, MutableAnyOrigin](input_buffer)

    input_host_buffer[0] = 1.0
    input_host_buffer[1] = 0.0
    input_host_buffer[2] = 1.0

    # Layer 1
    l1_weight_buffer = ctx.enqueue_create_buffer[DType.float32](l1_size * input_size)
    l1_weight_tensor = LayoutTensor[DType.float32, l1_weight_layout, MutableAnyOrigin](l1_weight_buffer)

    l1_bias_buffer = ctx.enqueue_create_buffer[DType.float32](l1_size)
    l1_bias_tensor = LayoutTensor[DType.float32, l1_weight_layout, MutableAnyOrigin](l1_bias_buffer)

    l1_out_buffer = ctx.enqueue_create_buffer[DType.float32](l1_size)
    l1_out_tensor = LayoutTensor[DType.float32, l1_out_layout, MutableAnyOrigin](l1_out_buffer)

    # Layer 2
    l2_weight_buffer = ctx.enqueue_create_buffer[DType.float32](l2_size * l1_size)
    l2_weight_tensor = LayoutTensor[DType.float32, l2_weight_layout, MutableAnyOrigin](l2_weight_buffer)

    l2_bias_buffer = ctx.enqueue_create_buffer[DType.float32](l2_size)
    l2_bias_tensor = LayoutTensor[DType.float32, l2_weight_layout, MutableAnyOrigin](l2_bias_buffer)

    l2_out_buffer = ctx.enqueue_create_buffer[DType.float32](l2_size)
    l2_out_tensor = LayoutTensor[DType.float32, l2_out_layout, MutableAnyOrigin](l2_out_buffer)

    # Layer 3
    l3_weight_buffer = ctx.enqueue_create_buffer[DType.float32](l3_size * l2_size)
    l3_weight_tensor = LayoutTensor[DType.float32, l3_weight_layout, MutableAnyOrigin](l3_weight_buffer)

    l3_bias_buffer = ctx.enqueue_create_buffer[DType.float32](l3_size)
    l3_bias_tensor = LayoutTensor[DType.float32, l3_weight_layout, MutableAnyOrigin](l3_bias_buffer)

    l3_out_buffer = ctx.enqueue_create_buffer[DType.float32](l3_size)
    l3_out_tensor = LayoutTensor[DType.float32, l3_out_layout, MutableAnyOrigin](l3_out_buffer)

    ctx.enqueue_copy(dst_buf=input_buffer, src_buf=input_host_buffer)

    ctx.enqueue_function[he_init[DType.float32, l1_weight_layout]
    ](
        l1_weight_tensor,
        43,
        grid_dim=(1),
        block_dim=(l1_draws),
    )

    ctx.enqueue_function[he_init[DType.float32, l2_weight_layout]
    ](
        l2_weight_tensor,
        43,
        grid_dim=(1),
        block_dim=(l2_draws),
    )

    ctx.enqueue_function[he_init[DType.float32, l3_weight_layout]
    ](
        l3_weight_tensor,
        43,
        grid_dim=(1),
        block_dim=(l3_draws),
    )

    ctx.synchronize()

    # layer 1
    ctx.enqueue_function[matmul[DType.float32, l1_weight_layout, input_layout, l1_out_layout]
    ](
        l1_weight_tensor,
        input_tensor,
        l1_out_tensor,
        grid_dim=(1),
        block_dim=(l1_weight_tensor.dim(0) * l1_weight_tensor.dim(1)),
    )

    ctx.synchronize()

    ctx.enqueue_function[vector_add[DType.float32, l1_out_layout]
    ](
        l1_out_tensor,
        l1_bias_tensor,
        grid_dim=(1),
        block_dim=(l1_out_tensor.dim(0)),
    )

    ctx.synchronize()

    ctx.enqueue_function[relu[DType.float32, l1_out_layout]
    ](
        l1_out_tensor,
        grid_dim=(1),
        block_dim=(l1_out_tensor.dim(0)),
    )

    ctx.synchronize()

    # layer 2
    ctx.enqueue_function[matmul[DType.float32, l2_weight_layout, l1_out_layout, l2_out_layout]
    ](
        l2_weight_tensor,
        l1_out_tensor,
        l2_out_tensor,
        grid_dim=(1),
        block_dim=(l2_weight_tensor.dim(0) * l2_weight_tensor.dim(1)),
    )

    ctx.synchronize()

    ctx.enqueue_function[vector_add[DType.float32, l2_out_layout]
    ](
        l2_out_tensor,
        l2_bias_tensor,
        grid_dim=(1),
        block_dim=(l2_out_tensor.dim(0)),
    )

    ctx.synchronize()

    ctx.enqueue_function[relu[DType.float32, l2_out_layout]
    ](
        l2_out_tensor,
        grid_dim=(1),
        block_dim=(l2_out_tensor.dim(0)),
    )

    ctx.synchronize()

    # layer 3
    ctx.enqueue_function[matmul[DType.float32, l3_weight_layout, l2_out_layout, l3_out_layout]
    ](
        l3_weight_tensor,
        l2_out_tensor,
        l3_out_tensor,
        grid_dim=(1),
        block_dim=(l3_weight_tensor.dim(0) * l3_weight_tensor.dim(1)),
    )

    ctx.synchronize()

    ctx.enqueue_function[vector_add[DType.float32, l3_out_layout]
    ](
        l3_out_tensor,
        l3_bias_tensor,
        grid_dim=(1),
        block_dim=(l3_out_tensor.dim(0)),
    )

    ctx.enqueue_function[tanh[DType.float32, l3_out_layout]
    ](
        l3_out_tensor,
        grid_dim=(1),
        block_dim=(l3_out_tensor.dim(0)),
    )

    ctx.synchronize()

    print('=== L3 out ===')
    print_matrix[l3_out_layout](ctx, l3_out_tensor, l3_out_buffer)

