from layout.layout_tensor import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer
from gpu.memory import AddressSpace
from gpu.cluster import cluster_arrive, cluster_wait
from gpu import block_dim, block_idx, thread_idx, barrier, block
from os import Atomic
from math import sqrt, tanh, exp
from time import monotonic
from gpu.random import NormalRandom
from print_utils import print_matrix_2, print_matrix_3, print_matrix_special
from he_init import he_init
from layer import DenseLayer, TPB
from activation import ReluActivation, TanhActivation


struct EuropeanCallLoss[num_inputs: Int, steps: Int, num_paths: Int]:

    var ctx: DeviceContext

    alias ResultTensorType = LayoutTensor[DType.float32, Layout.row_major(1),                            MutableAnyOrigin]
    alias GradTensorType   = LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutableAnyOrigin]
    alias InputTensorType  = LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin] 

    var result_buffer: DeviceBuffer[DType.float32]
    var grad_buffer:   DeviceBuffer[DType.float32]

    var result_tensor: Self.ResultTensorType
    var grad_tensor:   Self.GradTensorType

    var strike: Float32

    fn __init__(out self, ctx: DeviceContext, strike: Float32) raises:
        self.ctx = ctx
        self.strike = strike
        self.result_buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.grad_buffer   = ctx.enqueue_create_buffer[DType.float32](1 * steps * num_paths)
        self.result_tensor = Self.ResultTensorType(self.result_buffer)
        self.grad_tensor   = Self.GradTensorType(self.grad_buffer)

    fn print_loss(self) raises:
        print()
        print('==== loss  ====')
        print_matrix_2[1, 1](self.ctx, self.result_buffer)

    fn print_grad(self) raises:
        print()
        print('==== loss grad ====')
        print_matrix_special[1, steps, num_paths](self.ctx, self.grad_buffer)

    fn apply[layout: Layout](
            self, 
            input_tensor: LayoutTensor[DType.float32, layout, MutableAnyOrigin]
    ) raises:

        fn european_call_loss_kernel(
                input_tensor:  Self.InputTensorType,
                result_tensor: Self.ResultTensorType,
                strike: Float32,
        ):
            path = thread_idx.x

            if path >= num_paths:
                return

            value: Float32 = 0
            for step in range(1, steps):
                value += input_tensor[2, step, path][0] * (input_tensor[1, step, path] - input_tensor[1, step - 1, path])[0]

            payoff: Float32 = max(input_tensor[1, steps - 1, path][0] - strike, 0)
            error:  Float32 = (value - payoff) ** 2

            total_error = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](error))

            cluster_arrive()
            cluster_wait()

            if thread_idx.x == 0:
                result_tensor[0] = total_error

        self.ctx.enqueue_function_checked[european_call_loss_kernel, european_call_loss_kernel](
                input_tensor,
                self.result_tensor,
                self.strike,
                grid_dim=(1),
                block_dim=(num_paths),
        )
        self.ctx.synchronize()

    fn apply_grad(
            self,
            input_tensor: Self.InputTensorType,
    ) raises:

        fn european_call_grad_kernel(
                input_tensor: Self.InputTensorType,
                grad_tensor:  Self.GradTensorType,
                strike:       Float32,
        ):
            path = thread_idx.x

            if path >= num_paths:
                return

            value: Float32 = 0

            for step in range(steps - 1):
                value += input_tensor[2, step + 1, path] [0]* (input_tensor[1, step + 1, path][0] - input_tensor[1, step, path])[0]

            payoff = max(input_tensor[1, steps - 1, path] - strike, 0)
            grad_tensor[0, steps, path] = 0

            for step in range(steps - 1):
                grad_tensor[0, step, path] = 2 * (value - payoff) * (input_tensor[1, step + 1, path] - input_tensor[1, step, path])

        self.ctx.enqueue_function_checked[european_call_grad_kernel, european_call_grad_kernel](
                input_tensor,
                self.grad_tensor,
                self.strike,
                grid_dim=(1),
                block_dim=(num_paths),
        )
        self.ctx.synchronize()


fn generate_paths[num_inputs: Int, steps: Int, num_paths: Int](
        ctx: DeviceContext,
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
        drift:        Float32,
        vol:          Float32,
        seed:         Int,
) raises:

    fn generate_paths_kernel(
            input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
            drift:        Float32,
            vol:          Float32,
            seed:         Int,
    ):
        path = thread_idx.x

        if path >= num_paths or block_idx.x > 0:
            return

        input_tensor[0, 0, path] = 1
        input_tensor[1, 0, path] = 1
        input_tensor[2, 0, path] = 0

        dt = Float32(1.0 / (steps - 1))

        thread_seed = seed * block_dim.x + thread_idx.x
        random = NormalRandom(seed=thread_seed)

        for step in range(1, steps):
            if step < steps - 1:
                input_tensor[0, step, path] = 1 - dt * step
            else:
                input_tensor[0, step, path] = 0

            z = Float32(random.step_normal()[0])
            input_tensor[1, step, path] = input_tensor[1, step - 1, path] * exp((drift - 0.5 * vol ** 2)*dt + vol * sqrt(dt) * z)
            input_tensor[2, step, path] = 0

    ctx.enqueue_function_checked[generate_paths_kernel, generate_paths_kernel](
            input_tensor,
            drift,
            vol,
            seed,
            grid_dim=(1),
            block_dim=(num_paths),
    )
    ctx.synchronize()


fn next_step[input_size: Int, steps: Int, num_paths: Int](
        ctx: DeviceContext,
        output_tensor: LayoutTensor[DType.float32, Layout.row_major(1,          steps, num_paths), MutableAnyOrigin],
        input_tensor:  LayoutTensor[DType.float32, Layout.row_major(input_size, steps, num_paths), MutableAnyOrigin],
        step: Int
) raises:

    fn next_step_kernel(
        output_tensor: LayoutTensor[DType.float32, Layout.row_major(1,          steps, num_paths), MutableAnyOrigin],
        input_tensor:  LayoutTensor[DType.float32, Layout.row_major(input_size, steps, num_paths), MutableAnyOrigin],
        step: Int
    ):
        path = thread_idx.x

        if path >= num_paths or block_idx.x > 0:
            return

        input_tensor[2, step + 1, path] = output_tensor[0, step, path]

    ctx.enqueue_function_checked[next_step_kernel, next_step_kernel](
            output_tensor,
            input_tensor,
            step,
            grid_dim=(1),
            block_dim=(num_paths),
    )
    ctx.synchronize()


fn update_loss_grad[num_inputs: Int, steps: Int, num_paths: Int](
        ctx: DeviceContext,
        grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutableAnyOrigin],
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
        step:         Int,
) raises:

    fn update_loss_grad_kernel(
        grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutableAnyOrigin],
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
        step:         Int,
    ):
        path = thread_idx.x
        step_idx = block_idx.x

        if path >= num_paths or step >= steps:
            return

        if step_idx <= step:
            grad_tensor[0, step_idx, path] = input_tensor[2, step_idx + 1, path]
        else:
            grad_tensor[0, step_idx, path] = 0.0

    ctx.enqueue_function_checked[update_loss_grad_kernel, update_loss_grad_kernel](
            grad_tensor,
            input_tensor,
            step,
            grid_dim=(steps),
            block_dim=(num_paths),
    )
    ctx.synchronize()

fn main() raises:
    ctx = DeviceContext()

    print('Runnint on', ctx.name())
    print('MULTIPROCESSOR_COUNT                ', ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT))
    print('MAX_BLOCKS_PER_MULTIPROCESSOR       ', ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR))
    print('MAX_THREADS_PER_BLOCK               ', ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK))
    print('WARP_SIZE                           ', ctx.get_attribute(DeviceAttribute.WARP_SIZE))
    print('MAX_SHARED_MEMORY_PER_BLOCK         ', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK))
    print('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
    print()

    alias inputs       = 3
    alias steps        = 20
    alias network_size = 16
    alias num_paths    = 1024

    learning_rate: Float32 = 1e-4
    beta1:         Float32 = 0.9
    beta2:         Float32 = 0.999
    eps:           Float32 = 10e-8
    weight_decay:  Float32 = 0.01

    layer_1 = DenseLayer[network_size, inputs,       steps, num_paths, ReluActivation](ctx, 'layer 1', learning_rate, beta1, beta2, eps, weight_decay)
    layer_2 = DenseLayer[network_size, network_size, steps, num_paths, ReluActivation](ctx, 'layer 2', learning_rate, beta1, beta2, eps, weight_decay)
    layer_3 = DenseLayer[1,            network_size, steps, num_paths, TanhActivation](ctx, 'layer 3', learning_rate, beta1, beta2, eps, weight_decay)
    loss    = EuropeanCallLoss[inputs, steps, num_paths](ctx, 1.05)

    t = monotonic()
    for batch in range(1_000_000):
        generate_paths[inputs, steps, num_paths](ctx, layer_1.in_tensor, 0, 0.2, batch)

        for step in range(steps - 1):
            layer_1.apply(step)
            layer_1.feed_next(layer_2.in_tensor, step)
            layer_2.apply(step)
            layer_2.feed_next(layer_3.in_tensor, step)
            layer_3.apply(step)

            next_step[inputs, steps, num_paths](ctx, layer_3.out_tensor, layer_1.in_tensor, step)

        loss.apply(layer_1.in_tensor)
        loss.apply_grad(layer_1.in_tensor)

        for step in reversed(range(steps - 1)):
            layer_3.apply_grad(loss.grad_tensor)
            layer_2.apply_grad(layer_3.grad_tensor)
            layer_1.apply_grad(layer_2.grad_tensor)

            update_loss_grad[inputs, steps, num_paths](ctx, loss.grad_tensor, layer_1.grad_tensor, step)

        if batch % 100 == 0:
            print('batch', batch)
            loss.print_loss()

            layer_1.print_input(0)
            layer_1.print_input(1)
            layer_1.print_input(2)
            layer_1.print_input(3)
            layer_1.print_input(4)
            layer_1.print_input(5)
            layer_1.print_input(6)
            layer_1.print_input(7)

