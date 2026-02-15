from python import PythonObject
from time import monotonic
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer
from gpu import block_dim, block_idx, thread_idx, barrier
from gpu.primitives import block
from buffer import NDBuffer
from utils import IndexList
from random import NormalRandom
from math import exp, sqrt
from dense_layer import DenseLayer, TPB
from activation import ReluActivation, TanhActivation

comptime dtype = DType.float32

fn generate_paths_kernel(
        inputs:    Int,
        num_steps: Int,
        num_paths: Int,
        drift:     Float32,
        vol:       Float32,
        seed:      UInt64,
        ptr:       UnsafePointer[Float32, MutAnyOrigin],
):
    path = Int(thread_idx.x)

    x = NDBuffer[dtype, 3, MutAnyOrigin](ptr, IndexList[3](inputs, num_steps, num_paths))

    x[0, 0, path] = 1
    x[1, 0, path] = 1
    x[2, 0, path] = 0

    dt = 1.0 / Float32(num_steps - 1)

    thread_seed = seed * UInt64(block_dim.x + thread_idx.x)
    random = NormalRandom(seed=thread_seed)

    for step in range(1, num_steps):
        if step < num_steps - 1:
            x[0, step, path] = 1 - dt * Float32(step)
        else:
            x[0, step, path] = 0

        z = Float32(random.step_normal()[0])
        x[1, step, path] = x[1, step - 1, path] * exp((drift - 0.5 * vol ** 2)*dt + vol * sqrt(dt) * z)
        x[2, step, path] = 0

struct EuropeanCallLoss(Movable):

    var ctx: DeviceContext

    var inputs: Int
    var num_steps:  Int
    var num_paths:  Int

    var result_buffer: DeviceBuffer[dtype]
    var grad_buffer:   DeviceBuffer[dtype]

    var strike:   Float32
    var slippage: Float32

    fn __init__(
            out self, 
            ctx:        DeviceContext,
            inputs:     Int,
            num_steps:  Int,
            num_paths:  Int,
            strike:     Float32,
            slippage:   Float32,
    ) raises:
        self.ctx           = ctx
        self.inputs        = inputs
        self.num_steps     = num_steps
        self.num_paths     = num_paths
        self.strike        = strike
        self.slippage      = slippage
        self.result_buffer = ctx.enqueue_create_buffer[dtype](1)
        self.grad_buffer   = ctx.enqueue_create_buffer[dtype](1 * num_steps * num_paths)


    fn value(self) raises -> Float32:
        host_buffer = self.ctx.enqueue_create_host_buffer[dtype](1)
        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.result_buffer)
        self.ctx.synchronize()

        return host_buffer[0]

    fn fwd(self, input_buffer: DeviceBuffer) raises:

        fn fwd_kernel(
                inputs:     Int,
                num_steps:  Int,
                num_paths:  Int,
                x_ptr:      UnsafePointer[Float32, MutAnyOrigin],
                y_ptr:      UnsafePointer[Float32, MutAnyOrigin],
                strike:     Float32,
                slippage:   Float32,
        ):
            path = Int(thread_idx.x)

            x = NDBuffer[dtype, 3, MutAnyOrigin](x_ptr, IndexList[3](inputs, num_steps, num_paths))

            value: Float32 = 0

            for step in range(1, num_steps):

                d   = x[2, step,     path]
                s_c = x[1, step,     path]
                s_p = x[1, step - 1, path]

                value += d * (s_c * (1 + slippage) - s_p * (1 - slippage))

            payoff: Float32 = max(x[1, num_steps - 1, path] - strike, 0)
            error:  Float32 = (value - payoff) ** 2

            barrier()

            if thread_idx.x == 0:
                total_error = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](error))
                y_ptr[0] = total_error

        self.ctx.enqueue_function_experimental[fwd_kernel](
                self.inputs,
                self.num_steps,
                self.num_paths,
                input_buffer,
                self.result_buffer,
                self.strike,
                self.slippage,
                grid_dim=(1),
                block_dim=self.num_paths,
        )
        self.ctx.synchronize()

    fn bwd(self, input_buffer: DeviceBuffer) raises:

        fn bwd_kernel(
                inputs:     Int,
                num_steps:  Int,
                num_paths:  Int,
                x_ptr:      UnsafePointer[Float32, MutAnyOrigin],
                d_ptr:      UnsafePointer[Float32, MutAnyOrigin],
                strike:     Float32,
                slippage:   Float32,
        ):
            path = Int(thread_idx.x)

            x = NDBuffer[dtype, 3, MutAnyOrigin](x_ptr, IndexList[3](inputs, num_steps, num_paths))
            y = NDBuffer[dtype, 3, MutAnyOrigin](d_ptr, IndexList[3](1, num_steps, num_paths))

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

        self.ctx.enqueue_function_experimental[bwd_kernel](
                self.inputs,
                self.num_steps,
                self.num_paths,
                input_buffer,
                self.grad_buffer,
                self.strike,
                self.slippage,
                grid_dim=(1),
                block_dim=(self.num_paths),
        )
        self.ctx.synchronize()


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
        self.ctx = DeviceContext()

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
                params.inputs,
                params.num_steps,
                params.num_paths,
                params.strike,
                params.slippage,
        )

    fn __moveinit__(out self, deinit existing: Self):
        self.params = existing.params^
        self.ctx = existing.ctx^
        self.layer_1 = existing.layer_1^
        self.layer_2 = existing.layer_2^
        self.layer_3 = existing.layer_3^
        self.loss = existing.loss^

    fn generate_test_path(self) raises -> HostBuffer[dtype]:

        seed: UInt64 = UInt64(monotonic())
        comptime num_paths = 1

        buffer_size = self.params.inputs * self.params.num_steps
        host_buffer = self.ctx.enqueue_create_host_buffer[dtype](buffer_size)
        buffer      = self.ctx.enqueue_create_buffer[dtype](buffer_size)

        self.ctx.enqueue_function_experimental[generate_paths_kernel](
                self.params.inputs,
                self.params.num_steps,
                num_paths,
                self.params.drift,
                self.params.vol,
                seed,
                buffer,
                grid_dim=(1),
                block_dim=(num_paths),
        )

        self.ctx.synchronize()
        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=buffer)
        self.ctx.synchronize()

        return host_buffer

    fn generate_paths(self) raises:

        self.ctx.enqueue_function_experimental[generate_paths_kernel](
                self.params.inputs,
                self.params.num_steps,
                self.params.num_paths,
                self.params.drift,
                self.params.vol,
                self.params.seed,
                self.layer_1.x_buffer,
                grid_dim=(1),
                block_dim=(self.params.num_paths),
        )
        self.ctx.synchronize()

    fn copy_buffer(
            self,
            M:           Int,
            step:        Int,
            from_buffer: DeviceBuffer,
            to_buffer:   DeviceBuffer,
    ) raises:

        fn feed_next_kernel(
                num_steps: Int,
                num_paths: Int,
                M:         Int,
                step:      Int,
                from_ptr:  UnsafePointer[Float32, MutAnyOrigin],
                to_ptr:    UnsafePointer[Float32, MutAnyOrigin],
        ):
            path = Int(thread_idx.x)
            i = Int(block_idx.x)

            from_buffer = NDBuffer[dtype, 3, MutAnyOrigin](from_ptr, IndexList[3](M, num_steps, num_paths))
            to_buffer   = NDBuffer[dtype, 3, MutAnyOrigin](to_ptr,   IndexList[3](M, num_steps, num_paths))

            if path >= num_paths or i >= M:
                return

            to_buffer[i, step, path] = from_buffer[i, step, path]

        self.ctx.enqueue_function_experimental[feed_next_kernel](
                self.params.num_steps,
                self.params.num_paths,
                M,
                step,
                from_buffer,
                to_buffer,
                grid_dim=(M),
                block_dim=(self.params.num_paths),
        )
        self.ctx.synchronize()

    fn next_step(self, step: Int) raises:

        fn next_step_kernel(
                inputs:    Int,
                num_steps: Int,
                num_paths: Int,
                step:      Int,
                from_ptr:  UnsafePointer[Float32, MutAnyOrigin],
                to_ptr:    UnsafePointer[Float32, MutAnyOrigin],
        ):
            path = Int(thread_idx.x)

            from_buffer = NDBuffer[dtype, 3, MutAnyOrigin](from_ptr, IndexList[3](1,      num_steps, num_paths))
            to_buffer   = NDBuffer[dtype, 3, MutAnyOrigin](to_ptr,   IndexList[3](inputs, num_steps, num_paths))

            to_buffer[2, step + 1, path] = from_buffer[0, step, path]

        self.ctx.enqueue_function_experimental[next_step_kernel](
                self.params.inputs,
                self.params.num_steps,
                self.params.num_paths,
                step,
                self.layer_3.y_buffer, 
                self.layer_1.x_buffer,
                grid_dim=(1),
                block_dim=(self.params.num_paths),
        )
        self.ctx.synchronize()

    fn update_loss_grad(self, step: Int) raises:

        fn update_loss_grad_kernel(
                inputs:    Int,
                num_steps: Int,
                num_paths: Int,
                step:      Int,
                grad_ptr:  UnsafePointer[Float32, MutAnyOrigin],
                x_ptr:     UnsafePointer[Float32, MutAnyOrigin],
        ):
            path     = Int(thread_idx.x)
            step_idx = Int(block_idx.x)

            grad_buffer = NDBuffer[dtype, 3, MutAnyOrigin](grad_ptr, IndexList[3](1,      num_steps, num_paths))
            x_buffer    = NDBuffer[dtype, 3, MutAnyOrigin](x_ptr,    IndexList[3](inputs, num_steps, num_paths))

            if step_idx <= step:
                grad_buffer[0, step_idx, path] = x_buffer[2, step_idx + 1, path]
            else:
                grad_buffer[0, step_idx, path] = 0.0

        self.ctx.enqueue_function_experimental[update_loss_grad_kernel](
                self.params.inputs,
                self.params.num_steps,
                self.params.num_paths,
                step,
                self.loss.grad_buffer,
                self.layer_1.d_buffer,
                grid_dim=(self.params.num_steps),
                block_dim=(self.params.num_paths),
        )
        self.ctx.synchronize()


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

            # abort()

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
        tmp_buffer = self.ctx.enqueue_create_buffer[DType.float32](
                self.params.inputs * self.params.num_steps
        )
        self.ctx.enqueue_copy(dst_buf=tmp_buffer, src_buf=test_path)
        self.ctx.synchronize()
        self.layer_1.put_test_path(tmp_buffer)
        self.fwd(1)
        self.layer_1.get_test_path(tmp_buffer)
        self.ctx.enqueue_copy(dst_buf=test_path, src_buf=tmp_buffer)
        self.ctx.synchronize()

    fn loss_value(self) raises -> Float32:
        host_buffer = self.ctx.enqueue_create_host_buffer[dtype](1)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.loss.result_buffer)
        self.ctx.synchronize()

        return host_buffer[0]


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

    params = Params(
            inputs       = 3,
            network_size = 16,
            num_steps    = 20,
            num_paths    = 1024,
            lr           = 1e-3,
            lr_d1        = 0.95,
            lr_d2        = 10_000,
            beta1        = 0.9,
            beta2        = 0.999,
            eps          = 1e-8,
            weight_decay = 0.01,
            drift        = 0,
            vol          = 0.2,
            strike       = 1.1,
            slippage     = 0.01,
            seed         = 42,
    )

    network = Network(params)

    while True:
        @parameter
        for _ in range(100):
            network.run()
        print(network.loss_value())

