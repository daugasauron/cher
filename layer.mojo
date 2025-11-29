from layout.layout_tensor import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer
from gpu.memory import AddressSpace
from gpu.cluster import cluster_arrive, cluster_wait
from gpu import block_dim, block_idx, thread_idx, barrier, block
from os import Atomic
from he_init import he_init
from math import sqrt, tanh
from time import monotonic
from random import randn_float64, seed
from utils.numerics import isfinite
from print_utils import print_matrix_2, print_matrix_3, print_matrix_special

alias TPB = 1024

fn init_paths[num_inputs: Int, steps: Int, num_paths: Int](
    init_tensor:  LayoutTensor[DType.float32, Layout.row_major(steps,      num_paths, 2),         MutableAnyOrigin],
    input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps,     num_paths), MutableAnyOrigin],
):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= num_paths:
        return

    for i in range(steps):
        input_tensor[0, i, tid] = init_tensor[i, tid, 0]
        input_tensor[1, i, tid] = init_tensor[i, tid, 1]


fn recurrent_kernel[input_size: Int, network_size: Int, steps: Int, num_paths: Int](
    output_tensor: LayoutTensor[DType.float32, Layout.row_major(1,          steps, num_paths), MutableAnyOrigin],
    input_tensor:  LayoutTensor[DType.float32, Layout.row_major(input_size, steps, num_paths), MutableAnyOrigin],
    step: Int
):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid >= num_paths:
        return

    for i in range(num_paths):
        input_tensor[2, step + 1, tid] = output_tensor[0, step, tid]


struct DenseLayer[
        M:                  Int, 
        N:                  Int, 
        steps:              Int, 
        num_paths:          Int,
        activation_fn:      fn[width: Int](SIMD[DType.float32, width]) -> SIMD[DType.float32, width],
        activation_grad_fn: fn[width: Int](SIMD[DType.float32, width]) -> SIMD[DType.float32, width],
]:

    alias weight_layout:  Layout = Layout.row_major(M, N)
    alias bias_layout:    Layout = Layout.row_major(M)
    alias in_layout:      Layout = Layout.row_major(N, steps, num_paths)
    alias out_layout:     Layout = Layout.row_major(M, steps, num_paths)
    alias in_vec_layout:  Layout = Layout.row_major(N, num_paths)
    alias out_vec_layout: Layout = Layout.row_major(M, num_paths)

    alias WeightTensorType = LayoutTensor[DType.float32, Self.weight_layout,  MutableAnyOrigin]
    alias BiasTensorType   = LayoutTensor[DType.float32, Self.bias_layout,    MutableAnyOrigin]
    alias InTensorType     = LayoutTensor[DType.float32, Self.in_layout,      MutableAnyOrigin]
    alias OutTensorType    = LayoutTensor[DType.float32, Self.out_layout,     MutableAnyOrigin]

    alias random_seed: Int = 42
    alias random_size: Int = 4
    alias random_draws: Int = (N * M + Self.random_size - 1) // Self.random_size

    var ctx:           DeviceContext
    var layer_name:    String
    var learning_rate: Float32
    var beta1:         Float32
    var beta2:         Float32
    var eps:           Float32
    var weight_decay:  Float32

    var weight_buffer          : DeviceBuffer[DType.float32]
    var adams_weight_m1_buffer : DeviceBuffer[DType.float32]
    var adams_weight_m2_buffer : DeviceBuffer[DType.float32]
    var bias_buffer            : DeviceBuffer[DType.float32]
    var adams_bias_m1_buffer   : DeviceBuffer[DType.float32]
    var adams_bias_m2_buffer   : DeviceBuffer[DType.float32]
    var in_buffer              : DeviceBuffer[DType.float32]
    var out_buffer             : DeviceBuffer[DType.float32]
    var grad_buffer            : DeviceBuffer[DType.float32]

    var weight_tensor:          Self.WeightTensorType
    var adams_weight_m1_tensor: Self.WeightTensorType
    var adams_weight_m2_tensor: Self.WeightTensorType
    var bias_tensor:            Self.BiasTensorType
    var adams_bias_m1_tensor:   Self.BiasTensorType
    var adams_bias_m2_tensor:   Self.BiasTensorType
    var in_tensor:              Self.InTensorType
    var out_tensor:             Self.OutTensorType
    var grad_tensor:            Self.InTensorType

    var counter: Int

    fn __init__(
            out self, 
            ctx:           DeviceContext, 
            layer_name:    String, 
            learning_rate: Float32,
            beta1:         Float32,
            beta2:         Float32,
            eps:           Float32,
            weight_decay:  Float32,
    ) raises:
        self.ctx = ctx
        self.layer_name = layer_name
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.weight_buffer          = ctx.enqueue_create_buffer[DType.float32](M * N)
        self.adams_weight_m1_buffer = ctx.enqueue_create_buffer[DType.float32](M * N)
        self.adams_weight_m2_buffer = ctx.enqueue_create_buffer[DType.float32](M * N)
        self.bias_buffer            = ctx.enqueue_create_buffer[DType.float32](M)
        self.adams_bias_m1_buffer   = ctx.enqueue_create_buffer[DType.float32](M)
        self.adams_bias_m2_buffer   = ctx.enqueue_create_buffer[DType.float32](M)
        self.in_buffer              = ctx.enqueue_create_buffer[DType.float32](N * steps * num_paths)
        self.out_buffer             = ctx.enqueue_create_buffer[DType.float32](M * steps * num_paths)
        self.grad_buffer            = ctx.enqueue_create_buffer[DType.float32](N * steps * num_paths)

        self.weight_tensor          = Self.WeightTensorType(self.weight_buffer)
        self.adams_weight_m1_tensor = Self.WeightTensorType(self.adams_weight_m1_buffer)
        self.adams_weight_m2_tensor = Self.WeightTensorType(self.adams_weight_m2_buffer)
        self.bias_tensor            = Self.BiasTensorType(self.bias_buffer)
        self.adams_bias_m1_tensor   = Self.BiasTensorType(self.adams_bias_m1_buffer)
        self.adams_bias_m2_tensor   = Self.BiasTensorType(self.adams_bias_m2_buffer)
        self.in_tensor              = Self.InTensorType(self.in_buffer)
        self.out_tensor             = Self.OutTensorType(self.out_buffer)
        self.grad_tensor            = Self.InTensorType(self.grad_buffer)

        self.counter = 1

        ctx.enqueue_function[he_init[self.weight_layout]](
            self.weight_tensor,
            self.random_seed,
            grid_dim=(1),
            block_dim=(self.random_draws),
        )

    fn print_weights(self) raises:
        print()
        print('====', self.layer_name, 'weights ====')
        print_matrix_2[M, N](self.ctx, self.weight_buffer)


    fn print_input(self, path: Int) raises:
        print()
        print('====', self.layer_name, 'in, path:' , path, ' ====')
        print_matrix_3[N, steps, num_paths](self.ctx, self.in_buffer, path)

    fn print_bias(self) raises:
        print()
        print('====', self.layer_name, 'bias ====')
        print_matrix_2[M, 1](self.ctx, self.bias_buffer)

    fn print_output(self, path: Int) raises:
        print()
        print('====', self.layer_name, 'out, path:' , path, ' ====')
        print_matrix_3[M, steps, num_paths](self.ctx, self.out_buffer, path)

    fn print_grad(self, path: Int) raises:
        print()
        print('====', self.layer_name, 'grad, path:', path, ' ====')
        print_matrix_3[N, steps, num_paths](self.ctx, self.grad_buffer, path)

    fn apply(self, step: Int) raises:

        fn apply_kernel(
            weight_tensor: Self.WeightTensorType,
            bias_tensor:   Self.BiasTensorType,
            in_tensor:     Self.InTensorType,
            out_tensor:    Self.OutTensorType,
            step:          Int,
        ):
            var tid = block_idx.x * block_dim.x + thread_idx.x

            if tid >= num_paths:
                return

            for i in range(M):
                value: Float32 = 0
                for j in range(N):
                    value += weight_tensor[i, j][0] * in_tensor[j, step, tid][0]

                value += bias_tensor[i][0]
                value = activation_fn(value)

                out_tensor[i, step, tid] = value

        self.ctx.enqueue_function_checked[apply_kernel, apply_kernel](
            self.weight_tensor,
            self.bias_tensor,
            self.in_tensor,
            self.out_tensor,
            step,
            grid_dim=(1),
            block_dim=(num_paths),
        )
        self.ctx.synchronize()

    fn apply_grad(mut self, upstream_tensor: Self.OutTensorType) raises:

        fn downstream_kernel(
            weight_tensor:          Self.WeightTensorType,
            out_tensor:             Self.OutTensorType,
            upstream_tensor:        Self.OutTensorType,
            downstream_tensor:      Self.InTensorType,
        ):
            j    = block_idx.x
            step = block_idx.y
            path = thread_idx.x

            value: Float32 = 0
            for i in range(M):
                value += weight_tensor[i, j][0] * upstream_tensor[i, step, path][0] * activation_grad_fn(out_tensor[i, step, path])[0]
            downstream_tensor[j, step, path] = value

        fn path_sum_kernel(
            weight_tensor:          Self.WeightTensorType,
            bias_tensor:            Self.BiasTensorType,
            in_tensor:              Self.InTensorType,
            out_tensor:             Self.OutTensorType,
            upstream_tensor:        Self.OutTensorType,
            weight_temp_tensor:     LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutableAnyOrigin],
            bias_temp_tensor:       LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutableAnyOrigin],
        ):
            i    = block_idx.x
            j    = block_idx.y
            step = block_idx.z + 1
            path = thread_idx.x

            bias_update:   Float32 = upstream_tensor[i, step, path][0] * activation_grad_fn(out_tensor[i, step, path])[0]
            weight_update: Float32 = in_tensor[j, step, path][0] * bias_update

            weight_temp_tensor[i, j, step - 1] = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](weight_update))

            if j == 0:
                bias_temp_tensor[i, step - 1] = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](bias_update))

        fn update_kernel(
            weight_tensor:          Self.WeightTensorType,
            adams_weight_m1_tensor: Self.WeightTensorType,
            adams_weight_m2_tensor: Self.WeightTensorType,
            bias_tensor:            Self.BiasTensorType,
            adams_bias_m1_tensor:   Self.BiasTensorType,
            adams_bias_m2_tensor:   Self.BiasTensorType,
            weight_temp_tensor:     LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutableAnyOrigin],
            bias_temp_tensor:       LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutableAnyOrigin],
            counter:                Int,
            learning_rate:          Float32,
            beta1:                  Float32,
            beta2:                  Float32,
            eps:                    Float32,
            weight_decay:           Float32,
        ):
            i    = block_idx.x
            j    = block_idx.y
            step = thread_idx.x

            weight_update_t: Float32 = weight_temp_tensor[i, j, step][0]
            bias_update_t:   Float32 = bias_temp_tensor[i, step][0]

            weight_update = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](weight_update_t))
            bias_update   = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](bias_update_t))

            cluster_arrive()
            cluster_wait()

            weight_m1_old = adams_weight_m1_tensor[i, j]
            weight_m2_old = adams_weight_m2_tensor[i, j]

            weight_m1_new = beta1 * weight_m1_old + (1 - beta1) * weight_update
            weight_m2_new = beta2 * weight_m2_old + (1 - beta2) * weight_update * weight_update

            weight_m1_hat = weight_m1_new / (1 - pow(beta1, counter))
            weight_m2_hat = weight_m2_new / (1 - pow(beta2, counter))

            weight_tensor[i, j] -= learning_rate * weight_decay * weight_tensor[i, j]
            weight_tensor[i, j] -= learning_rate * weight_m1_hat / (sqrt(weight_m2_hat) + eps)

            adams_weight_m1_tensor[i, j] = weight_m1_new
            adams_weight_m2_tensor[i, j] = weight_m2_new

            if j == 0:
                m1_old = adams_bias_m1_tensor[i]
                m2_old = adams_bias_m2_tensor[i]

                m1_new = beta1 * m1_old + (1 - beta1) * bias_update
                m2_new = beta2 * m2_old + (1 - beta2) * bias_update * bias_update

                m1_hat = m1_new / (1 - pow(beta1, counter))
                m2_hat = m2_new / (1 - pow(beta2, counter))

                bias_tensor[i] -= learning_rate * m1_hat / (sqrt(m2_hat) + eps)

                adams_bias_m1_tensor[i] = m1_new
                adams_bias_m2_tensor[i] = m2_new


        weight_temp_buffer = self.ctx.enqueue_create_buffer[DType.float32](M * N * steps - 1)
        bias_temp_buffer   = self.ctx.enqueue_create_buffer[DType.float32](M * steps - 1)

        weight_temp_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutableAnyOrigin](weight_temp_buffer)
        bias_temp_tensor   = LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutableAnyOrigin](bias_temp_buffer)

        self.ctx.enqueue_function_checked[downstream_kernel, downstream_kernel](
            self.weight_tensor,
            self.out_tensor,
            upstream_tensor,
            self.grad_tensor,
            grid_dim=(N, steps - 1),
            block_dim=(num_paths),
        )

        self.ctx.enqueue_function_checked[path_sum_kernel, path_sum_kernel](
            self.weight_tensor,
            self.bias_tensor,
            self.in_tensor,
            self.out_tensor,
            upstream_tensor,
            weight_temp_tensor,
            bias_temp_tensor,
            grid_dim=(M, N, steps - 1),
            block_dim=(num_paths),
        )

        self.ctx.synchronize()

        self.ctx.enqueue_function_checked[update_kernel, update_kernel](
            self.weight_tensor,
            self.adams_weight_m1_tensor,
            self.adams_weight_m2_tensor,
            self.bias_tensor,
            self.adams_bias_m1_tensor,
            self.adams_bias_m2_tensor,
            weight_temp_tensor,
            bias_temp_tensor,
            self.counter,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.eps,
            self.weight_decay,
            grid_dim=(M, N),
            block_dim=(steps - 1),
        )

        self.ctx.synchronize()

        self.counter += 1

    fn feed_next(
            self,
            next_in_tensor: Self.OutTensorType,
            step: Int,
    ) raises:
        fn feed_next_kernel(
            output_tensor: Self.OutTensorType,
            input_tensor:  Self.OutTensorType,
            step: Int
        ):
            var tid = block_idx.x * block_dim.x + thread_idx.x

            if tid >= M:
                return

            for i in range(num_paths):
                input_tensor[tid, step, i] = output_tensor[tid, step, i]

        self.ctx.enqueue_function_checked[feed_next_kernel, feed_next_kernel](
            self.out_tensor,
            next_in_tensor,
            step,
            grid_dim=(1),
            block_dim=(M),
        )
        self.ctx.synchronize()


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
            var tid = block_idx.x * block_dim.x + thread_idx.x

            if tid > 0:
                return

            result_tensor[0] = 0

            var value: Float32 
            for i in range(num_paths):
                value = 0
                for j in range(1, steps):
                    value += input_tensor[2, j, i][0] * (input_tensor[1, j, i] - input_tensor[1, j - 1, i])[0]

                payoff = max(input_tensor[1, steps - 1, i] - strike, 0)
                result_tensor[0] += (value - payoff) ** 2

        self.ctx.enqueue_function_checked[european_call_loss_kernel, european_call_loss_kernel](
            input_tensor,
            self.result_tensor,
            self.strike,
            grid_dim=(1),
            block_dim=(1),
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
            var tid = block_idx.x * block_dim.x + thread_idx.x

            if tid > 0:
                return

            var value: Float32

            for i in range(num_paths):

                value = 0

                for j in range(steps - 1):
                    value += input_tensor[2, j + 1, i] [0]* (input_tensor[1, j + 1, i][0] - input_tensor[1, j, i])[0]

                payoff = max(input_tensor[1, steps - 1, i] - strike, 0)
                grad_tensor[0, steps, i] = 0
                for j in range(steps - 1):
                    grad_tensor[0, j, i] = 2 * (value - payoff) * (input_tensor[1, j + 1, i] - input_tensor[1, j, i])

        self.ctx.enqueue_function_checked[european_call_grad_kernel, european_call_grad_kernel](
            input_tensor,
            self.grad_tensor,
            self.strike,
            grid_dim=(1),
            block_dim=(1),
        )
        self.ctx.synchronize()


fn relu_act[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return max(y, 0)


fn relu_act_grad[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    if y > 0:
        return 1
    else:
        return 0


fn tanh_act[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return tanh(y)


fn tanh_act_grad[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return 1 - y ** 2


fn update_loss_grad_kernel[num_inputs: Int, steps: Int, num_paths: Int](
    grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutableAnyOrigin],
    input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
    step:         Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid > 0:
        return

    for i in range(steps):
        for j in range(num_paths):
            if i <= step:
                grad_tensor[0, i, j] = input_tensor[2, i + 1, j]
            else:
                grad_tensor[0, i, j] = 0.0

fn yeah(msg: String, time: UInt) -> UInt:
    var t = monotonic()
    print(msg, 'elapsed (ms):', (t - time) / 1_000_000)
    return t

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

    var learning_rate: Float32 = 1e-4
    var beta1:         Float32 = 0.9
    var beta2:         Float32 = 0.999
    var eps:           Float32 = 10e-8
    var weight_decay:  Float32 = 0.01

    layer_1 = DenseLayer[network_size, inputs,       steps, num_paths, relu_act, relu_act_grad](ctx, 'layer 1', learning_rate, beta1, beta2, eps, weight_decay)
    layer_2 = DenseLayer[network_size, network_size, steps, num_paths, relu_act, relu_act_grad](ctx, 'layer 2', learning_rate, beta1, beta2, eps, weight_decay)
    layer_3 = DenseLayer[1,            network_size, steps, num_paths, tanh_act, tanh_act_grad](ctx, 'layer 3', learning_rate, beta1, beta2, eps, weight_decay)
    loss    = EuropeanCallLoss[inputs, steps, num_paths](ctx, 1.05)

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
            host_init_tensor[j, i, 1] = host_init_tensor[j - 1, i, 1] * (1 + Float32(randn_float64(0, 0.05)))

    ctx.enqueue_copy(dst_buf=device_init_buffer, src_buf=host_init_buffer)
    ctx.synchronize()

    ctx.enqueue_function[init_paths[inputs, steps, num_paths]](
        device_init_tensor,
        layer_1.in_tensor,
        grid_dim=(1),
        block_dim=(num_paths),
    )
    ctx.synchronize()

    var t = monotonic()
    for batch in range(1_000_000):

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

        for step in reversed(range(steps - 1)):
            layer_3.apply_grad(loss.grad_tensor)
            layer_2.apply_grad(layer_3.grad_tensor)
            layer_1.apply_grad(layer_2.grad_tensor)

            ctx.enqueue_function[update_loss_grad_kernel[inputs, steps, num_paths]](
                loss.grad_tensor,
                layer_1.grad_tensor,
                step,
                grid_dim=(1),
                block_dim=(1),
            )
            ctx.synchronize()

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

            t = yeah('batch', t)


