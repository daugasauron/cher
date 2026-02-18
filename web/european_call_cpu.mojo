from python import PythonObject
from time import monotonic
from gpu.host import DeviceContext, HostBuffer
from buffer import NDBuffer
from utils import IndexList
from random import NormalRandom
from math import exp, sqrt
from dense_layer_cpu import DenseLayer
from activation import ReluActivation, TanhActivation

comptime dtype: DType = DType.float32


struct EuropeanCallLoss(Movable):

    var ctx: DeviceContext
    var params_ptr: UnsafePointer[Params, MutAnyOrigin]

    var result_buffer: HostBuffer[dtype]
    var grad_buffer:   HostBuffer[dtype]

    fn __init__(
            out self,
            ctx:      DeviceContext,
            params:   UnsafePointer[Params, MutAnyOrigin],
    ) raises:
        self.ctx           = ctx
        self.params_ptr    = params
        self.result_buffer = ctx.enqueue_create_host_buffer[dtype](1)
        self.grad_buffer   = ctx.enqueue_create_host_buffer[dtype](1 * params[].num_steps * params[].num_paths)

        print('inputs', params[].inputs)
        print('num_steps', params[].num_steps)
        print('num_paths', params[].num_paths)

    fn value(self) raises -> Float32:
        return self.result_buffer[0]

    fn fwd(self, input_buffer: HostBuffer[dtype]) raises:
        inputs    = self.params_ptr[].inputs
        num_steps = self.params_ptr[].num_steps
        num_paths = self.params_ptr[].num_paths
        strike    = self.params_ptr[].strike
        slippage  = self.params_ptr[].slippage

        x = NDBuffer[dtype, 3, MutAnyOrigin](
                input_buffer.unsafe_ptr(),
                IndexList[3](inputs, num_steps, num_paths),
        )

        total_error: Float32 = 0

        for path in range(num_paths):
            value: Float32 = 0

            for step in range(1, num_steps):
                d   = x[2, step,     path]
                s_c = x[1, step,     path]
                s_p = x[1, step - 1, path]

                value += d * (s_c * (1 + slippage) - s_p * (1 - slippage))

            payoff: Float32 = max(x[1, num_steps - 1, path] - strike, 0)
            error:  Float32 = (value - payoff) ** 2
            total_error += error

        self.result_buffer.unsafe_ptr()[0] = total_error

    fn bwd(self, input_buffer: HostBuffer[dtype]) raises:
        inputs    = self.params_ptr[].inputs
        num_steps = self.params_ptr[].num_steps
        num_paths = self.params_ptr[].num_paths
        strike    = self.params_ptr[].strike
        slippage  = self.params_ptr[].slippage

        x = NDBuffer[dtype, 3, MutAnyOrigin](
                input_buffer.unsafe_ptr(),
                IndexList[3](inputs, num_steps, num_paths),
        )
        y = NDBuffer[dtype, 3, MutAnyOrigin](
                self.grad_buffer.unsafe_ptr(),
                IndexList[3](1, num_steps, num_paths),
        )

        for path in range(num_paths):
            value: Float32 = 0

            for step in range(1, num_steps):
                d   = x[2, step,     path]
                s_c = x[1, step,     path]
                s_p = x[1, step - 1, path]

                value += d * (s_c * (1 + slippage) - s_p * (1 - slippage))

            payoff = max(x[1, num_steps - 1, path] - strike, 0)

            y[0, num_steps, path] = 0

            for step in range(1, num_steps):
                s_c = x[1, step - 1, path]
                s_n = x[1, step,     path]

                y[0, step - 1, path] = 2 * (value - payoff) * (s_n * (1 + slippage) - s_c * (1 - slippage))


@fieldwise_init
struct Params(ImplicitlyCopyable):

    var inputs:       Int
    var network_size: Int
    var num_steps:    Int
    var num_paths:    Int

    var lr:           Float32
    var lr_d1:        Float32
    var lr_d2:        Float32
    var beta1:        Float32
    var beta2:        Float32
    var eps:          Float32
    var weight_decay: Float32

    var drift:    Float32
    var vol:      Float32
    var strike:   Float32
    var slippage: Float32

    var seed: UInt64


struct Network(Movable):

    var params: Params
    var ctx:    DeviceContext

    var layer_1: DenseLayer[ReluActivation]
    var layer_2: DenseLayer[ReluActivation]
    var layer_3: DenseLayer[TanhActivation]
    var loss:    EuropeanCallLoss

    def __init__(out self, params: Params):
        self.params = params
        self.ctx = DeviceContext(api="cpu")
        print('Running on', self.ctx.name())

        self.layer_1 = DenseLayer[ReluActivation](
                self.ctx,
                params.network_size,
                params.inputs,
                params.num_steps,
                params.num_paths,
                params.lr,
                params.lr_d1,
                params.lr_d2,
                params.beta1,
                params.beta2,
                params.eps,
                params.weight_decay,
                params.seed,
        )

        self.layer_2 = DenseLayer[ReluActivation](
                self.ctx,
                params.network_size,
                params.network_size,
                params.num_steps,
                params.num_paths,
                params.lr,
                params.lr_d1,
                params.lr_d2,
                params.beta1,
                params.beta2,
                params.eps,
                params.weight_decay,
                params.seed,
        )

        self.layer_3 = DenseLayer[TanhActivation](
                self.ctx,
                1,
                params.network_size,
                params.num_steps,
                params.num_paths,
                params.lr,
                params.lr_d1,
                params.lr_d2,
                params.beta1,
                params.beta2,
                params.eps,
                params.weight_decay,
                params.seed,
        )

        self.loss = EuropeanCallLoss(
                self.ctx,
                UnsafePointer(to=self.params),
        )

    fn __moveinit__(out self, deinit existing: Self):
        self.params = existing.params^
        self.ctx = existing.ctx^
        self.layer_1 = existing.layer_1^
        self.layer_2 = existing.layer_2^
        self.layer_3 = existing.layer_3^
        self.loss = existing.loss^
        # Update the params pointer to point to the new location
        self.loss.params_ptr = UnsafePointer(to=self.params)

    fn generate_test_path(self) raises -> HostBuffer[dtype]:

        seed: UInt64 = UInt64(monotonic())
        comptime num_paths = 1

        buffer_size = self.params.inputs * self.params.num_steps
        host_buffer = self.ctx.enqueue_create_host_buffer[dtype](buffer_size)

        x = NDBuffer[dtype, 3, MutAnyOrigin](
                host_buffer.unsafe_ptr(),
                IndexList[3](self.params.inputs, self.params.num_steps, num_paths),
        )

        # Generate single path
        x[0, 0, 0] = 1
        x[1, 0, 0] = 1
        x[2, 0, 0] = 0

        dt = 1.0 / Float32(self.params.num_steps - 1)

        thread_seed = seed * UInt64(num_paths + 0)
        random = NormalRandom(seed=thread_seed)

        for step in range(1, self.params.num_steps):
            if step < self.params.num_steps - 1:
                x[0, step, 0] = 1 - dt * Float32(step)
            else:
                x[0, step, 0] = 0

            z = Float32(random.step_normal()[0])
            x[1, step, 0] = x[1, step - 1, 0] * exp((self.params.drift - 0.5 * self.params.vol ** 2)*dt + self.params.vol * sqrt(dt) * z)
            x[2, step, 0] = 0

        return host_buffer

    fn generate_paths(self) raises:
        x = NDBuffer[dtype, 3, MutAnyOrigin](
                self.layer_1.x_buffer.unsafe_ptr(),
                IndexList[3](self.params.inputs, self.params.num_steps, self.params.num_paths),
        )

        dt = 1.0 / Float32(self.params.num_steps - 1)

        for path in range(self.params.num_paths):
            x[0, 0, path] = 1
            x[1, 0, path] = 1
            x[2, 0, path] = 0

            thread_seed = self.params.seed * UInt64(self.params.num_paths + path)
            random = NormalRandom(seed=thread_seed)

            for step in range(1, self.params.num_steps):
                if step < self.params.num_steps - 1:
                    x[0, step, path] = 1 - dt * Float32(step)
                else:
                    x[0, step, path] = 0

                z = Float32(random.step_normal()[0])
                x[1, step, path] = x[1, step - 1, path] * exp((self.params.drift - 0.5 * self.params.vol ** 2)*dt + self.params.vol * sqrt(dt) * z)
                x[2, step, path] = 0

    fn copy_buffer(
            self,
            M:           Int,
            step:        Int,
            from_buffer: HostBuffer[dtype],
            to_buffer:   HostBuffer[dtype],
    ) raises:
        from_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                from_buffer.unsafe_ptr(),
                IndexList[3](M, self.params.num_steps, self.params.num_paths),
        )
        to_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                to_buffer.unsafe_ptr(),
                IndexList[3](M, self.params.num_steps, self.params.num_paths),
        )

        for i in range(M):
            for path in range(self.params.num_paths):
                to_buf[i, step, path] = from_buf[i, step, path]

    fn next_step(self, step: Int) raises:
        from_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                self.layer_3.y_buffer.unsafe_ptr(),
                IndexList[3](1, self.params.num_steps, self.params.num_paths),
        )
        to_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                self.layer_1.x_buffer.unsafe_ptr(),
                IndexList[3](self.params.inputs, self.params.num_steps, self.params.num_paths),
        )

        for path in range(self.params.num_paths):
            to_buf[2, step + 1, path] = from_buf[0, step, path]

    fn update_loss_grad(self, step: Int) raises:
        grad_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                self.loss.grad_buffer.unsafe_ptr(),
                IndexList[3](1, self.params.num_steps, self.params.num_paths),
        )
        x_buf = NDBuffer[dtype, 3, MutAnyOrigin](
                self.layer_1.d_buffer.unsafe_ptr(),
                IndexList[3](self.params.inputs, self.params.num_steps, self.params.num_paths),
        )

        for step_idx in range(self.params.num_steps):
            for path in range(self.params.num_paths):
                if step_idx <= step:
                    grad_buf[0, step_idx, path] = x_buf[2, step_idx + 1, path]
                else:
                    grad_buf[0, step_idx, path] = 0.0

    fn fwd(self) raises:
        self.fwd(self.params.num_paths)

    fn fwd(self, num_paths: Int) raises:

        for step in range(self.params.num_steps - 1):

            self.layer_1.fwd(step)
            self.copy_buffer(self.params.network_size, step, self.layer_1.y_buffer, self.layer_2.x_buffer)

            self.layer_2.fwd(step)
            self.copy_buffer(self.params.network_size, step, self.layer_2.y_buffer, self.layer_3.x_buffer)

            self.layer_3.fwd(step)
            self.next_step(step)

        self.loss.fwd(self.layer_1.x_buffer)


    fn bwd(mut self) raises:

        self.loss.bwd(self.layer_1.x_buffer)

        for step in reversed(range(self.params.num_steps - 1)):
            self.layer_3.bwd(self.loss.grad_buffer)
            self.layer_2.bwd(self.layer_3.d_buffer)
            self.layer_1.bwd(self.layer_2.d_buffer)
            self.update_loss_grad(step)

    fn run(mut self) raises:
        self.generate_paths()
        self.fwd()
        self.bwd()

    fn run_test(self, test_path: HostBuffer[dtype]) raises:
        self.layer_1.put_test_path(test_path)
        self.fwd(1)
        self.layer_1.get_test_path(test_path)

    fn loss_value(self) raises -> Float32:
        return self.loss.result_buffer[0]

    fn reset_counters(mut self):
        self.layer_1.reset_counter()
        self.layer_2.reset_counter()
        self.layer_3.reset_counter()
