from gpu.host import DeviceContext
from layout.layout_tensor import Layout, LayoutTensor, IntTuple
from random import seed
from layer import (
        DenseLayer,
        EuropeanCallLoss,
        relu_activation,
        relu_activation_grad,
        tanh_activation,
        tanh_activation_grad,
        init_paths,
        recurrent_kernel,
        update_loss_grad_kernel,
)

fn main() raises:
    ctx = DeviceContext()

    alias inputs       = 3
    alias steps        = 3
    alias network_size = 4
    alias num_paths    = 2

    var learning_rate: Float32 = 100

    layer_1 = DenseLayer[network_size, inputs,       steps, num_paths, relu_activation, relu_activation_grad](ctx, 'layer 1')
    layer_2 = DenseLayer[network_size, network_size, steps, num_paths, relu_activation, relu_activation_grad](ctx, 'layer 2')
    layer_3 = DenseLayer[1,            network_size, steps, num_paths, tanh_activation, tanh_activation_grad](ctx, 'layer 3')

    loss = EuropeanCallLoss[inputs, steps, num_paths](ctx, 1.05)

    var host_init_buffer   = ctx.enqueue_create_host_buffer[DType.float32](num_paths * steps * 2)
    var device_init_buffer = ctx.enqueue_create_buffer[DType.float32](num_paths * steps * 2)

    var host_init_tensor   = LayoutTensor[DType.float32, Layout.row_major(steps, num_paths, 2), MutableAnyOrigin](host_init_buffer)
    var device_init_tensor = LayoutTensor[DType.float32, Layout.row_major(steps, num_paths, 2), MutableAnyOrigin](device_init_buffer)

    var step: Float32 = 1.0 / Float32(steps - 1)

    # Init first step
    for i in range(num_paths):
        host_init_tensor[0, i, 0] = 1
        host_init_tensor[0, i, 1] = 1

    seed(42)

    for i in range(num_paths):
        for j in range(1, steps):
            host_init_tensor[j, i, 0] = 1.0 - step * Float32(j)

    host_init_tensor[1, 0, 1] = 1.1
    host_init_tensor[2, 0, 1] = 1.2
    host_init_tensor[1, 1, 1] = 0.9
    host_init_tensor[2, 1, 1] = 0.8

    ctx.enqueue_copy(dst_buf=device_init_buffer, src_buf=host_init_buffer)
    ctx.synchronize()

    ctx.enqueue_function[init_paths[inputs, steps, num_paths]](
        device_init_tensor,
        layer_1.in_tensor,
        grid_dim=(1),
        block_dim=(num_paths),
    )
    ctx.synchronize()

    layer_1.print_weights()
    layer_1.print_bias()
    layer_1.print_input(0)
    layer_1.print_output(0)

    for batch in range(1):
        for step in range(steps - 1):
            layer_1.apply(step)
            layer_1.feed_next(layer_2.in_tensor, step)

            layer_2.apply(step)
            layer_2.feed_next(layer_3.in_tensor, step)

            layer_3.apply(step)

            ctx.enqueue_function[recurrent_kernel[inputs, network_size, steps, num_paths]](
                layer_3.out_tensor,
                layer_1.in_tensor,
                step,
                grid_dim=(1),
                block_dim=(num_paths),
            )
            ctx.synchronize()

        loss.apply(layer_1.in_tensor)
        loss.apply_grad(layer_1.in_tensor)

        loss.print_grad()

        for step in reversed(range(1, steps)):
            layer_3.apply_grad(loss.grad_tensor, learning_rate)
            print()
            print('## step', step)
            layer_3.print_grad(1)
            layer_2.apply_grad(layer_3.grad_tensor, learning_rate)
            layer_1.apply_grad(layer_2.grad_tensor, learning_rate)

            ctx.enqueue_function[update_loss_grad_kernel[inputs, steps, num_paths]](
                loss.grad_tensor,
                layer_1.in_tensor,
                step,
                grid_dim=(1),
                block_dim=(1),
            )


