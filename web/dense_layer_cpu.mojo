from math import sqrt
from utils import IndexList
from random import NormalRandom
from buffer import NDBuffer
from activation import Activation
from gpu.host import DeviceContext, HostBuffer

comptime dtype: DType = DType.float32


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

    var weight_buffer:         HostBuffer[dtype]
    var weight_adam_m1_buffer: HostBuffer[dtype]
    var weight_adam_m2_buffer: HostBuffer[dtype]

    var bias_buffer:           HostBuffer[dtype]
    var bias_adam_m1_buffer:   HostBuffer[dtype]
    var bias_adam_m2_buffer:   HostBuffer[dtype]

    var x_buffer: HostBuffer[dtype]
    var y_buffer: HostBuffer[dtype]
    var d_buffer: HostBuffer[dtype]

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

        self.weight_buffer         = ctx.enqueue_create_host_buffer[dtype](M * N)
        self.weight_adam_m1_buffer = ctx.enqueue_create_host_buffer[dtype](M * N)
        self.weight_adam_m2_buffer = ctx.enqueue_create_host_buffer[dtype](M * N)

        self.bias_buffer           = ctx.enqueue_create_host_buffer[dtype](M)
        self.bias_adam_m1_buffer   = ctx.enqueue_create_host_buffer[dtype](M)
        self.bias_adam_m2_buffer   = ctx.enqueue_create_host_buffer[dtype](M)

        self.x_buffer              = ctx.enqueue_create_host_buffer[dtype](N * num_steps * num_paths)
        self.d_buffer              = ctx.enqueue_create_host_buffer[dtype](N * num_steps * num_paths)
        self.y_buffer              = ctx.enqueue_create_host_buffer[dtype](M * num_steps * num_paths)

        # He initialization
        w = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=self.weight_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](M, N),
        )
        for i in range(M):
            for j in range(N):
                thread_seed = seed + UInt64(i * N + j)
                random = NormalRandom(seed=thread_seed)
                w[i, j] = ((2.0 / Float32(N)) ** 0.5) * random.step_normal()[0]

        self.counter = 1

    fn fwd(self, step: Int) raises:
        self.fwd(step, self.num_paths)

    fn fwd(self, step: Int, num_paths: Int) raises:
        w = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=self.weight_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.M, self.N),
        )
        b = NDBuffer[dtype, 1, MutAnyOrigin](
                ptr=self.bias_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[1](self.M),
        )
        x = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.x_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.N, self.num_steps, num_paths),
        )
        y = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.y_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.M, self.num_steps, num_paths),
        )

        for i in range(self.M):
            for path in range(num_paths):
                value: Float32 = 0
                for j in range(self.N):
                    value += w[i, j] * x[j, step, path]
                value += b[i]
                value = Self.activation.apply(value)
                y[i, step, path] = value

    fn bwd(mut self, upstream_buffer: HostBuffer[dtype]) raises:
        w = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=self.weight_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.M, self.N),
        )
        y = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.y_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.M, self.num_steps, self.num_paths),
        )
        d = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.d_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.N, self.num_steps, self.num_paths),
        )
        u = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=upstream_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.M, self.num_steps, self.num_paths),
        )
        x = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.x_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.N, self.num_steps, self.num_paths),
        )
        b = NDBuffer[dtype, 1, MutAnyOrigin](
                ptr=self.bias_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[1](self.M),
        )
        w_m1 = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=self.weight_adam_m1_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.M, self.N),
        )
        w_m2 = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=self.weight_adam_m2_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.M, self.N),
        )
        b_m1 = NDBuffer[dtype, 1, MutAnyOrigin](
                ptr=self.bias_adam_m1_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[1](self.M),
        )
        b_m2 = NDBuffer[dtype, 1, MutAnyOrigin](
                ptr=self.bias_adam_m2_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[1](self.M),
        )

        # Downstream: compute gradients w.r.t. input
        for j in range(self.N):
            for step in range(self.num_steps - 1):
                for path in range(self.num_paths):
                    value: Float32 = 0
                    for i in range(self.M):
                        value += w[i, j] * u[i, step, path] * Self.activation.apply_grad(y[i, step, path])
                    d[j, step, path] = value

        # Path sum: accumulate weight and bias gradients across paths and steps
        decayed_lr = self.lr * Float32(pow(self.lr_d1, Float32(self.counter) / self.lr_d2))

        for i in range(self.M):
            b_update_total: Float32 = 0

            for j in range(self.N):
                w_update_total: Float32 = 0

                for step in range(1, self.num_steps):
                    w_update_step: Float32 = 0
                    b_update_step: Float32 = 0

                    for path in range(self.num_paths):
                        b_update = u[i, step, path] * Self.activation.apply_grad(y[i, step, path])
                        w_update_step += x[j, step, path] * b_update
                        if j == 0:
                            b_update_step += b_update

                    w_update_total += w_update_step
                    if j == 0:
                        b_update_total += b_update_step

                # Adam weight update
                w_m1_old = w_m1[i, j]
                w_m2_old = w_m2[i, j]

                w_m1_new = self.beta1 * w_m1_old + (1 - self.beta1) * w_update_total
                w_m2_new = self.beta2 * w_m2_old + (1 - self.beta2) * w_update_total * w_update_total

                w_m1_hat = w_m1_new / (1 - pow(self.beta1, self.counter))
                w_m2_hat = w_m2_new / (1 - pow(self.beta2, self.counter))

                w[i, j] -= decayed_lr * self.weight_decay * w[i, j]
                w[i, j] -= decayed_lr * w_m1_hat / (sqrt(w_m2_hat) + self.eps)

                w_m1[i, j] = w_m1_new
                w_m2[i, j] = w_m2_new

            # Adam bias update
            m1_old = b_m1[i]
            m2_old = b_m2[i]

            m1_new = self.beta1 * m1_old + (1 - self.beta1) * b_update_total
            m2_new = self.beta2 * m2_old + (1 - self.beta2) * b_update_total * b_update_total

            m1_hat = m1_new / (1 - pow(self.beta1, self.counter))
            m2_hat = m2_new / (1 - pow(self.beta2, self.counter))

            b[i] -= decayed_lr * m1_hat / (sqrt(m2_hat) + self.eps)

            b_m1[i] = m1_new
            b_m2[i] = m2_new

        self.counter += 1

    fn reset_counter(mut self):
        self.counter = 1

    fn put_test_path(self, test_path_buffer: HostBuffer[dtype]) raises:
        x = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.x_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.N, self.num_steps, self.num_paths),
        )
        t = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=test_path_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.N, self.num_steps),
        )

        for i in range(self.N):
            for step in range(self.num_steps):
                x[i, step, 0] = t[i, step]

    fn get_test_path(self, test_path_buffer: HostBuffer[dtype]) raises:
        x = NDBuffer[dtype, 3, MutAnyOrigin](
                ptr=self.x_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[3](self.N, self.num_steps, self.num_paths),
        )
        t = NDBuffer[dtype, 2, MutAnyOrigin](
                ptr=test_path_buffer.unsafe_ptr(),
                dynamic_shape=IndexList[2](self.N, self.num_steps),
        )

        for i in range(self.N):
            for step in range(self.num_steps):
                t[i, step] = x[i, step, 0]
