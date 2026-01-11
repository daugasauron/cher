from layout.layout_tensor import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer
from gpu.cluster import cluster_arrive, cluster_wait
from gpu import block_dim, block_idx, thread_idx, block
from math import sqrt
from he_init import he_init
from print_utils import print_matrix_2, print_matrix_3, print_matrix_special
from activation import Activation, ReluActivation, TanhActivation

alias TPB = 1024

struct DenseLayer[
        M:          Int, 
        N:          Int, 
        steps:      Int, 
        num_paths:  Int,
        activation: Activation,
]:

    alias weight_layout:  Layout = Layout.row_major(M, N)
    alias bias_layout:    Layout = Layout.row_major(M)
    alias in_layout:      Layout = Layout.row_major(N, steps, num_paths)
    alias out_layout:     Layout = Layout.row_major(M, steps, num_paths)
    alias in_vec_layout:  Layout = Layout.row_major(N, num_paths)
    alias out_vec_layout: Layout = Layout.row_major(M, num_paths)

    alias WeightTensorType = LayoutTensor[DType.float32, Self.weight_layout,  MutAnyOrigin]
    alias BiasTensorType   = LayoutTensor[DType.float32, Self.bias_layout,    MutAnyOrigin]
    alias InTensorType     = LayoutTensor[DType.float32, Self.in_layout,      MutAnyOrigin]
    alias OutTensorType    = LayoutTensor[DType.float32, Self.out_layout,     MutAnyOrigin]

    alias random_seed: Int = 42
    alias random_size: Int = 4
    alias random_draws: Int = (N * M + Self.random_size - 1) // Self.random_size

    var ctx:                   DeviceContext
    var layer_name:            String
    var learning_rate:         Float32
    var learning_rate_decay_1: Float32
    var learning_rate_decay_2: Float32
    var beta1:                 Float32
    var beta2:                 Float32
    var eps:                   Float32
    var weight_decay:          Float32

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
            ctx:                   DeviceContext, 
            layer_name:            String, 
            learning_rate:         Float32,
            learning_rate_decay_1: Float32,
            learning_rate_decay_2: Float32,
            beta1:                 Float32,
            beta2:                 Float32,
            eps:                   Float32,
            weight_decay:          Float32,
    ) raises:
        self.ctx = ctx
        self.layer_name = layer_name
        self.learning_rate = learning_rate
        self.learning_rate_decay_1 = learning_rate_decay_1
        self.learning_rate_decay_2 = learning_rate_decay_2
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

        ctx.enqueue_function_checked[he_init[self.weight_layout], he_init[self.weight_layout]](
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
            i = Int(block_idx.x)
            path = Int(thread_idx.x)

            if path >= num_paths or i >= M:
                return

            value: Float32 = 0
            for j in range(N):
                value += weight_tensor[i, j][0] * in_tensor[j, step, path][0]

            value += bias_tensor[i][0]
            value = activation.apply(value)

            out_tensor[i, step, path] = value

        self.ctx.enqueue_function_checked[apply_kernel, apply_kernel](
            self.weight_tensor,
            self.bias_tensor,
            self.in_tensor,
            self.out_tensor,
            step,
            grid_dim=(M),
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
                value += weight_tensor[i, j][0] * upstream_tensor[i, step, path][0] * activation.apply_grad(out_tensor[i, step, path][0])
            downstream_tensor[j, step, path] = value

        fn path_sum_kernel(
            weight_tensor:          Self.WeightTensorType,
            bias_tensor:            Self.BiasTensorType,
            in_tensor:              Self.InTensorType,
            out_tensor:             Self.OutTensorType,
            upstream_tensor:        Self.OutTensorType,
            weight_temp_tensor:     LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutAnyOrigin],
            bias_temp_tensor:       LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutAnyOrigin],
        ):
            i    = block_idx.x
            j    = block_idx.y
            step = block_idx.z + 1
            path = thread_idx.x

            bias_update:   Float32 = upstream_tensor[i, step, path][0] * activation.apply_grad(out_tensor[i, step, path][0])
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
            weight_temp_tensor:     LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutAnyOrigin],
            bias_temp_tensor:       LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutAnyOrigin],
            counter:                Int,
            learning_rate:          Float32,
            learning_rate_decay_1:  Float32,
            learning_rate_decay_2:  Float32,
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

            decayed_learning_rate  = learning_rate * Float32(pow(learning_rate_decay_1, counter / learning_rate_decay_2))

            weight_m1_old = adams_weight_m1_tensor[i, j]
            weight_m2_old = adams_weight_m2_tensor[i, j]

            weight_m1_new = beta1 * weight_m1_old + (1 - beta1) * weight_update
            weight_m2_new = beta2 * weight_m2_old + (1 - beta2) * weight_update * weight_update

            weight_m1_hat = weight_m1_new / (1 - pow(beta1, counter))
            weight_m2_hat = weight_m2_new / (1 - pow(beta2, counter))

            weight_tensor[i, j] -= decayed_learning_rate * weight_decay * weight_tensor[i, j]
            weight_tensor[i, j] -= decayed_learning_rate * weight_m1_hat / (sqrt(weight_m2_hat) + eps)

            adams_weight_m1_tensor[i, j] = weight_m1_new
            adams_weight_m2_tensor[i, j] = weight_m2_new

            if j == 0:
                m1_old = adams_bias_m1_tensor[i]
                m2_old = adams_bias_m2_tensor[i]

                m1_new = beta1 * m1_old + (1 - beta1) * bias_update
                m2_new = beta2 * m2_old + (1 - beta2) * bias_update * bias_update

                m1_hat = m1_new / (1 - pow(beta1, counter))
                m2_hat = m2_new / (1 - pow(beta2, counter))

                bias_tensor[i] -= decayed_learning_rate * m1_hat / (sqrt(m2_hat) + eps)

                adams_bias_m1_tensor[i] = m1_new
                adams_bias_m2_tensor[i] = m2_new


        weight_temp_buffer = self.ctx.enqueue_create_buffer[DType.float32](M * N * steps - 1)
        bias_temp_buffer   = self.ctx.enqueue_create_buffer[DType.float32](M * steps - 1)

        weight_temp_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N, steps - 1), MutAnyOrigin](weight_temp_buffer)
        bias_temp_tensor   = LayoutTensor[DType.float32, Layout.row_major(M,    steps - 1), MutAnyOrigin](bias_temp_buffer)

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
            self.learning_rate_decay_1,
            self.learning_rate_decay_2,
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
            path = Int(thread_idx.x)
            i = Int(block_idx.x)

            if path >= num_paths or i >= M:
                return

            input_tensor[i, step, path] = output_tensor[i, step, path]

        self.ctx.enqueue_function_checked[feed_next_kernel, feed_next_kernel](
                self.out_tensor,
                next_in_tensor,
                step,
                grid_dim=(M),
                block_dim=(num_paths),
        )
        self.ctx.synchronize()

