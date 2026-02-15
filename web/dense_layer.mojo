from math import sqrt
from utils import IndexList
from random import NormalRandom
from buffer import NDBuffer
from activation import Activation
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import block_idx, thread_idx, cluster_arrive, cluster_wait
from gpu.primitives import block

comptime TPB = 1024
comptime dtype: DType = DType.float32


fn he_init_kernel(
        M:         Int,
        N:         Int,
        seed:      UInt64,
        ptr:       UnsafePointer[Float32, MutAnyOrigin],
):
    comptime rank = 2

    buffer = NDBuffer[dtype, rank, MutAnyOrigin](ptr, IndexList[rank](M, N))

    i    = Int(block_idx.x)
    j    = Int(block_idx.y)

    random = NormalRandom(seed=seed)

    buffer[i, j] = (Float32(2 / UInt64(N)) ** 0.5) * random.step_normal()[0]


struct DenseLayer[activation: Activation](Movable):

    var ctx: DeviceContext

    var M:         Int
    var N:         Int
    var num_steps: Int
    var num_paths: Int

    var lr:           Float32
    var lr_d1:        Float32
    var lr_d2:        Float32
    var beta1:        Float32
    var beta2:        Float32
    var eps:          Float32
    var weight_decay: Float32

    var seed: UInt64

    var weight_buffer:         DeviceBuffer[dtype]
    var weight_adam_m1_buffer: DeviceBuffer[dtype]
    var weight_adam_m2_buffer: DeviceBuffer[dtype]

    var bias_buffer:           DeviceBuffer[dtype]
    var bias_adam_m1_buffer:   DeviceBuffer[dtype]
    var bias_adam_m2_buffer:   DeviceBuffer[dtype]

    var x_buffer: DeviceBuffer[dtype]
    var y_buffer: DeviceBuffer[dtype]
    var d_buffer: DeviceBuffer[dtype]

    var counter: Int

    def __init__(
            out self, 
            ctx:          DeviceContext,
            M:            Int, 
            N:            Int, 
            num_steps:    Int, 
            num_paths:    Int,
            lr:           Float32,
            lr_d1:        Float32,
            lr_d2:        Float32,
            beta1:        Float32,
            beta2:        Float32,
            eps:          Float32,
            weight_decay: Float32,
            seed:         UInt64,
    ):
        self.ctx = ctx

        self.M =         M
        self.N =         N
        self.num_steps = num_steps
        self.num_paths = num_paths

        self.lr           = lr
        self.lr_d1        = lr_d1
        self.lr_d2        = lr_d2
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay

        self.seed = seed

        self.weight_buffer         = ctx.enqueue_create_buffer[dtype](M * N)
        self.weight_adam_m1_buffer = ctx.enqueue_create_buffer[dtype](M * N)
        self.weight_adam_m2_buffer = ctx.enqueue_create_buffer[dtype](M * N)

        self.bias_buffer           = ctx.enqueue_create_buffer[dtype](M)
        self.bias_adam_m1_buffer   = ctx.enqueue_create_buffer[dtype](M)
        self.bias_adam_m2_buffer   = ctx.enqueue_create_buffer[dtype](M)

        self.x_buffer              = ctx.enqueue_create_buffer[dtype](N * num_steps * num_paths)
        self.d_buffer              = ctx.enqueue_create_buffer[dtype](N * num_steps * num_paths)
        self.y_buffer              = ctx.enqueue_create_buffer[dtype](M * num_steps * num_paths)

        self.ctx.enqueue_function_experimental[he_init_kernel](
                self.M,
                self.N,
                self.seed,
                self.weight_buffer,
                grid_dim=(self.M, self.N),
                block_dim=self.num_paths,
        )

        self.counter = 1

    fn fwd(self, step: Int) raises:
        self.fwd(step, self.num_paths)

    fn fwd(self, step: Int, num_paths: Int) raises:

        fn fwd_kernel(
                M:         Int,
                N:         Int,
                num_steps: Int,
                num_paths: Int,
                w_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                b_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                x_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                y_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                step:      Int,
        ):
            i    = Int(block_idx.x)
            path = Int(thread_idx.x)

            w = NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_ptr, dynamic_shape=IndexList[2](M, N))
            b = NDBuffer[dtype, 1, MutAnyOrigin](ptr=b_ptr, dynamic_shape=IndexList[1](M))
            x = NDBuffer[dtype, 3, MutAnyOrigin](ptr=x_ptr, dynamic_shape=IndexList[3](N, num_steps, num_paths))
            y = NDBuffer[dtype, 3, MutAnyOrigin](ptr=y_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))

            value: Float32 = 0

            for j in range(N):
                value += w[i, j] * x[j, step, path]

            value += b[i]
            value = Self.activation.apply(value)

            y[i, step, path] = value

        self.ctx.enqueue_function_experimental[fwd_kernel](
                self.M,
                self.N,
                self.num_steps,
                num_paths,
                self.weight_buffer,
                self.bias_buffer,
                self.x_buffer,
                self.y_buffer,
                step,
                grid_dim=self.M,
                block_dim=num_paths,
        )
        self.ctx.synchronize()

    fn bwd(mut self, upstream_buffer: DeviceBuffer) raises:

        fn downstream_kernel(
                M:         Int,
                N:         Int,
                num_steps: Int,
                num_paths: Int,
                w_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                y_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                d_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                u_ptr:     UnsafePointer[Float32, MutAnyOrigin],
        ):
            j    = Int(block_idx.x)
            step = Int(block_idx.y)
            path = Int(thread_idx.x)

            w = NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_ptr, dynamic_shape=IndexList[2](M, N))
            y = NDBuffer[dtype, 3, MutAnyOrigin](ptr=y_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))
            d = NDBuffer[dtype, 3, MutAnyOrigin](ptr=d_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))
            u = NDBuffer[dtype, 3, MutAnyOrigin](ptr=u_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))

            value: Float32 = 0

            for i in range(M):
                value += w[i, j] * u[i, step, path] * Self.activation.apply_grad(y[i, step, path])

            d[j, step, path] = value

        fn path_sum_kernel(
                M:         Int,
                N:         Int,
                num_steps: Int,
                num_paths: Int,
                w_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                b_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                x_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                y_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                w_tmp_ptr: UnsafePointer[Float32, MutAnyOrigin],
                b_tmp_ptr: UnsafePointer[Float32, MutAnyOrigin],
                u_ptr:     UnsafePointer[Float32, MutAnyOrigin],

        ):
            i =    Int(block_idx.x)
            j =    Int(block_idx.y)
            step = Int(block_idx.z + 1)
            path = Int(thread_idx.x)

            w =     NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_ptr, dynamic_shape=IndexList[2](M, N))
            b =     NDBuffer[dtype, 1, MutAnyOrigin](ptr=b_ptr, dynamic_shape=IndexList[1](M))
            x =     NDBuffer[dtype, 3, MutAnyOrigin](ptr=x_ptr, dynamic_shape=IndexList[3](N, num_steps, num_paths))
            y =     NDBuffer[dtype, 3, MutAnyOrigin](ptr=y_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))
            u =     NDBuffer[dtype, 3, MutAnyOrigin](ptr=u_ptr, dynamic_shape=IndexList[3](M, num_steps, num_paths))
            w_tmp = NDBuffer[dtype, 3, MutAnyOrigin](ptr=w_tmp_ptr, dynamic_shape=IndexList[3](M, N, num_steps - 1))
            b_tmp = NDBuffer[dtype, 2, MutAnyOrigin](ptr=b_tmp_ptr, dynamic_shape=IndexList[2](M, num_steps - 1))

            b_update = u[i, step, path] * Self.activation.apply_grad(y[i, step, path])
            w_update = x[j, step, path] * b_update

            w_tmp[i, j, step - 1] = block.sum[block_size=TPB, broadcast=False](val=SIMD[dtype, 1](w_update))

            if j == 0:
                b_tmp[i, step - 1] = block.sum[block_size=TPB, broadcast=False](val=SIMD[dtype, 1](b_update))

        fn update_kernel(
                M:            Int,
                N:            Int,
                num_steps:    Int,
                num_paths:    Int,
                w_ptr:        UnsafePointer[Float32, MutAnyOrigin],
                w_m1_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                w_m2_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                b_ptr:        UnsafePointer[Float32, MutAnyOrigin],
                b_m1_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                b_m2_ptr:     UnsafePointer[Float32, MutAnyOrigin],
                w_tmp_ptr:    UnsafePointer[Float32, MutAnyOrigin],
                b_tmp_ptr:    UnsafePointer[Float32, MutAnyOrigin],
                counter:      Int,
                lr:           Float32,
                lr_d1:        Float32,
                lr_d2:        Float32,
                beta1:        Float32,
                beta2:        Float32,
                eps:          Float32,
                weight_decay: Float32,
        ):
            i    = Int(block_idx.x)
            j    = Int(block_idx.y)
            step = Int(thread_idx.x)

            w =     NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_ptr, dynamic_shape=IndexList[2](M, N))
            w_m1 =  NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_m1_ptr, dynamic_shape=IndexList[2](M, N))
            w_m2 =  NDBuffer[dtype, 2, MutAnyOrigin](ptr=w_m2_ptr, dynamic_shape=IndexList[2](M, N))
            b =     NDBuffer[dtype, 1, MutAnyOrigin](ptr=b_ptr, dynamic_shape=IndexList[1](M))
            b_m1 =  NDBuffer[dtype, 1, MutAnyOrigin](ptr=b_m1_ptr, dynamic_shape=IndexList[1](M))
            b_m2 =  NDBuffer[dtype, 1, MutAnyOrigin](ptr=b_m2_ptr, dynamic_shape=IndexList[1](M))
            w_tmp = NDBuffer[dtype, 3, MutAnyOrigin](ptr=w_tmp_ptr, dynamic_shape=IndexList[3](M, N, num_steps - 1))
            b_tmp = NDBuffer[dtype, 2, MutAnyOrigin](ptr=b_tmp_ptr, dynamic_shape=IndexList[2](M, num_steps - 1))

            w_update_t = w_tmp[i, j, step]
            b_update_t = b_tmp[i, step]

            w_update = block.sum[block_size=TPB, broadcast=False](val=SIMD[dtype, 1](w_update_t))
            b_update = block.sum[block_size=TPB, broadcast=False](val=SIMD[dtype, 1](b_update_t))

            cluster_arrive()
            cluster_wait()

            decayed_lr  = lr * Float32(pow(lr_d1, Float32(counter) / lr_d2))

            w_m1_old = w_m1[i, j]
            w_m2_old = w_m2[i, j]

            w_m1_new = beta1 * w_m1_old + (1 - beta1) * w_update
            w_m2_new = beta2 * w_m2_old + (1 - beta2) * w_update * w_update

            w_m1_hat = w_m1_new / (1 - pow(beta1, counter))
            w_m2_hat = w_m2_new / (1 - pow(beta2, counter))

            w[i, j] -= decayed_lr * weight_decay * w[i, j]
            w[i, j] -= decayed_lr * w_m1_hat / (sqrt(w_m2_hat) + eps)

            w_m1[i, j] = w_m1_new
            w_m2[i, j] = w_m2_new

            if j == 0:
                m1_old = b_m1[i]
                m2_old = b_m2[i]

                m1_new = beta1 * m1_old + (1 - beta1) * b_update
                m2_new = beta2 * m2_old + (1 - beta2) * b_update * b_update

                m1_hat = m1_new / (1 - pow(beta1, counter))
                m2_hat = m2_new / (1 - pow(beta2, counter))

                b[i] -= decayed_lr * m1_hat / (sqrt(m2_hat) + eps)

                b_m1[i] = m1_new
                b_m2[i] = m2_new

        w_tmp_buffer = self.ctx.enqueue_create_buffer[DType.float32](self.M * self.N * (self.num_steps - 1))
        b_tmp_buffer = self.ctx.enqueue_create_buffer[DType.float32](self.M * (self.num_steps - 1))

        self.ctx.enqueue_function_experimental[downstream_kernel](
                self.M,
                self.N,
                self.num_steps,
                self.num_paths,
                self.weight_buffer,
                self.y_buffer,
                self.d_buffer,
                upstream_buffer,
                grid_dim=(self.N, self.num_steps - 1),
                block_dim=self.num_paths,
        )

        self.ctx.enqueue_function_experimental[path_sum_kernel](
                self.M,
                self.N,
                self.num_steps,
                self.num_paths,
                self.weight_buffer,
                self.bias_buffer,
                self.x_buffer,
                self.y_buffer,
                w_tmp_buffer,
                b_tmp_buffer,
                upstream_buffer,
                grid_dim=(self.M, self.N, self.num_steps - 1),
                block_dim=self.num_paths,
        )

        self.ctx.synchronize()

        self.ctx.enqueue_function_experimental[update_kernel](
                self.M,
                self.N,
                self.num_steps,
                self.num_paths,
                self.weight_buffer,
                self.weight_adam_m1_buffer,
                self.weight_adam_m2_buffer,
                self.bias_buffer,
                self.bias_adam_m1_buffer,
                self.bias_adam_m2_buffer,
                w_tmp_buffer,
                b_tmp_buffer,
                self.counter,
                self.lr,
                self.lr_d1,
                self.lr_d2,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                grid_dim=(self.M, self.N),
                block_dim=self.num_steps - 1,
        )

        self.ctx.synchronize()

        self.counter += 1

    fn put_test_path(self, test_path_buffer: DeviceBuffer) raises:

        fn put_test_path_kernel(
                N:             Int,
                num_steps:     Int,
                num_paths:     Int,
                test_path_ptr: UnsafePointer[Float32, MutAnyOrigin],
                x_ptr:         UnsafePointer[Float32, MutAnyOrigin],
        ):
            i    = Int(block_idx.x)
            step = Int(block_idx.y)

            x = NDBuffer[dtype, 3, MutAnyOrigin](
                    ptr=x_ptr, 
                    dynamic_shape=IndexList[3](N, num_steps, num_paths)
            )

            t = NDBuffer[dtype, 2, MutAnyOrigin](
                    ptr=test_path_ptr, 
                    dynamic_shape=IndexList[2](N, num_steps)
            )

            x[i, step, 0] = t[i, step]

        self.ctx.enqueue_function_experimental[put_test_path_kernel](
                self.N,
                self.num_steps,
                self.num_paths,
                test_path_buffer,
                self.x_buffer,
                grid_dim=(self.N, self.num_steps),
                block_dim=(1),
        )
        self.ctx.synchronize()

    fn get_test_path(self, test_path_buffer: DeviceBuffer) raises:

        fn get_test_path_kernel(
                N:             Int,
                num_steps:     Int,
                num_paths:     Int,
                test_path_ptr: UnsafePointer[Float32, MutAnyOrigin],
                x_ptr:         UnsafePointer[Float32, MutAnyOrigin],
        ):
            i    = Int(block_idx.x)
            step = Int(block_idx.y)

            x = NDBuffer[dtype, 3, MutAnyOrigin](
                    ptr=x_ptr, 
                    dynamic_shape=IndexList[3](N, num_steps, num_paths)
            )

            t = NDBuffer[dtype, 2, MutAnyOrigin](
                    ptr=test_path_ptr, 
                    dynamic_shape=IndexList[2](N, num_steps)
            )
            t[i, step] = x[i, step, 0]

        self.ctx.enqueue_function_experimental[get_test_path_kernel](
                self.N,
                self.num_steps,
                self.num_paths,
                test_path_buffer,
                self.x_buffer,
                grid_dim=(self.N, self.num_steps),
                block_dim=(1),
        )
        self.ctx.synchronize()


fn main() raises:
    from activation import ReluActivation

    ctx = DeviceContext()

    layer = DenseLayer[ReluActivation](
            ctx,
            16,
            3,
            20,
            1024,
            1e-3,
            0.99,
            10_000,
            0.9,
            0.999,
            1e-8,
            0.01,
            42,
    )

