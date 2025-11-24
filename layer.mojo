from layout.layout_tensor import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_dim, block_idx, thread_idx
from os import Atomic
from he_init import he_init
from math import sqrt, tanh
from time import monotonic
from random import randn_float64, seed
from print_utils import print_matrix_2, print_matrix_3, print_matrix_special

alias SIMDFloat32 = SIMD[DType.float32, Layout.__init__(IntTuple[__origin_of()](1), IntTuple[__origin_of()](1)).size()]

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

    var layer_name: String
    var ctx: DeviceContext

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

    fn __init__(out self, ctx: DeviceContext, layer_name: String) raises:
        self.ctx = ctx
        self.layer_name = layer_name

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

    fn apply(
            self,
            step: Int,
    ) raises:

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
                value: SIMDFloat32 = 0
                for j in range(N):
                    value += weight_tensor[i, j] * in_tensor[j, step, tid]

                value += bias_tensor[i]
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

    fn apply_grad(
            mut self,
            upstream_tensor: Self.OutTensorType,
            learning_rate:   Float32,
            beta1:           Float32,
            beta2:           Float32,
            eps:             Float32,
            weight_decay:    Float32,
    ) raises:

        fn apply_grad_kernel(
            weight_tensor:          Self.WeightTensorType,
            adams_weight_m1_tensor: Self.WeightTensorType,
            adams_weight_m2_tensor: Self.WeightTensorType,
            bias_tensor:            Self.BiasTensorType,
            adams_bias_m1_tensor:   Self.BiasTensorType,
            adams_bias_m2_tensor:   Self.BiasTensorType,
            in_tensor:              Self.InTensorType,
            out_tensor:             Self.OutTensorType,
            upstream_tensor:        Self.OutTensorType,
            downstream_tensor:      Self.InTensorType,
            counter:                Int,
            learning_rate:          Float32,
            beta1:                  Float32,
            beta2:                  Float32,
            eps:                    Float32,
            weight_decay:           Float32,
        ):
            var tid = block_idx.x * block_dim.x + thread_idx.x

            if tid >= num_paths:
                return

            if tid < M:
                bias_update: SIMDFloat32 = 0
                for path in range(num_paths):
                    for step in range(1, steps):
                        bias_update += upstream_tensor[tid, step, path] * activation_grad_fn(out_tensor[tid, step, path])

                bias_m1_old = adams_bias_m1_tensor[tid]
                bias_m2_old = adams_bias_m2_tensor[tid]

                bias_m1_new = beta1 * bias_m1_old + (1 - beta1) * bias_update
                bias_m2_new = beta2 * bias_m2_old + (1 - beta2) * bias_update * bias_update

                bias_m1_hat = bias_m1_new / (1 - pow(beta1, counter))
                bias_m2_hat = bias_m2_new / (1 - pow(beta2, counter))

                bias_tensor[tid] -= learning_rate * bias_m1_hat / (sqrt(bias_m2_hat) + eps)

                adams_bias_m1_tensor[tid] = bias_m1_new
                adams_bias_m2_tensor[tid] = bias_m2_new

            if tid < M * N:
                weight_update: SIMDFloat32 = 0
                var i = tid // N
                var j = tid %  N

                for path in range(num_paths):
                    for step in range(1, steps):
                         weight_update += in_tensor[j, step, path] * upstream_tensor[i, step, path] * activation_grad_fn(out_tensor[i, step, path])

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

            # Downstream
            if tid < num_paths:
                for i in range(N):
                    for step in range(steps - 1):
                        value: SIMDFloat32 = 0
                        for j in range(M):
                            value += weight_tensor[j, i] * upstream_tensor[j, step, tid] * activation_grad_fn(out_tensor[j, step, tid])
                        downstream_tensor[i, step, tid] = value

        self.ctx.enqueue_function_checked[apply_grad_kernel, apply_grad_kernel](
            self.weight_tensor,
            self.adams_weight_m1_tensor,
            self.adams_weight_m2_tensor,
            self.bias_tensor,
            self.adams_bias_m1_tensor,
            self.adams_bias_m2_tensor,
            self.in_tensor,
            self.out_tensor,
            upstream_tensor,
            self.grad_tensor,
            self.counter,
            learning_rate,
            beta1,
            beta2,
            eps,
            weight_decay,
            grid_dim=(1),
            block_dim=(num_paths),
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

            var value: SIMDFloat32
            for i in range(num_paths):
                value = 0
                for j in range(1, steps):
                    value += input_tensor[2, j, i] * (input_tensor[1, j, i] - input_tensor[1, j - 1, i])

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

            var value: SIMDFloat32

            for i in range(num_paths):

                value = 0

                for j in range(steps - 1):
                    value += input_tensor[2, j + 1, i] * (input_tensor[1, j + 1, i] - input_tensor[1, j, i])

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


fn relu_activation[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return max(y, 0)


fn relu_activation_grad[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    if y > 0:
        return 1
    else:
        return 0


fn tanh_activation[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return tanh(y)


fn tanh_activation_grad[width: Int](y: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
    return 1 - y ** 2


fn update_loss_grad_kernel[num_inputs: Int, steps: Int, num_paths: Int](
    grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutableAnyOrigin],
    input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutableAnyOrigin],
    step:         Int,
):
    var tid = block_idx.x * block_dim.x + thread_idx.x

    if tid > 0:
        return

    for i in range(0, steps):
        for j in range(num_paths):
            if i < step:
                grad_tensor[0, i, j] = input_tensor[2, i, j]
            else:
                grad_tensor[0, i, j] = 0.0

fn yeah(msg: String, time: UInt) -> UInt:
    var t = monotonic()
    print(msg, 'elapsed (ms):', (t - time) / 1_000_000)
    return t

fn main() raises:
    ctx = DeviceContext()

    alias inputs       = 3
    alias steps        = 20
    alias network_size = 16
    alias num_paths    = 1024

    var learning_rate: Float32 = 1e-3
    var beta1:         Float32 = 0.9
    var beta2:         Float32 = 0.999
    var eps:           Float32 = 10e-8
    var weight_decay:  Float32 = 0.01

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

        for step in reversed(range(1, steps)):
            layer_3.apply_grad(loss.grad_tensor,    learning_rate, beta1, beta2, eps, weight_decay)
            layer_2.apply_grad(layer_3.grad_tensor, learning_rate, beta1, beta2, eps, weight_decay)
            layer_1.apply_grad(layer_2.grad_tensor, learning_rate, beta1, beta2, eps, weight_decay)

            ctx.enqueue_function[update_loss_grad_kernel[inputs, steps, num_paths]](
                loss.grad_tensor,
                layer_1.grad_tensor,
                step,
                grid_dim=(1),
                block_dim=(1),
            )
            ctx.synchronize()

        if batch % 10 == 0:
            print('batch', batch)
            loss.print_loss()

            layer_1.print_input(0)
            layer_1.print_input(1)
            layer_1.print_input(2)
            layer_1.print_input(3)
            layer_1.print_input(4)
            layer_1.print_input(5)

            t = yeah('batch', t)


