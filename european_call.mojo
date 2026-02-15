from python import Python
from layout.layout_tensor import Layout, LayoutTensor, IntTuple
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer
from gpu.memory import AddressSpace
from gpu.primitives import cluster_arrive, cluster_wait, block
from gpu import block_dim, block_idx, thread_idx, barrier
from math import sqrt, tanh, exp
from random import NormalRandom
from time import monotonic
from print_utils import print_matrix_2, print_matrix_3, print_matrix_special
from he_init import he_init
from layer import DenseLayer, TPB
from activation import ReluActivation, TanhActivation


struct EuropeanCallLoss[num_inputs: Int, steps: Int, num_paths: Int]:

    var ctx: DeviceContext

    comptime ResultTensorType = LayoutTensor[DType.float32, Layout.row_major(1),                                           MutAnyOrigin]
    comptime GradTensorType   = LayoutTensor[DType.float32, Layout.row_major(1, Self.steps, Self.num_paths),               MutAnyOrigin]
    comptime InputTensorType  = LayoutTensor[DType.float32, Layout.row_major(Self.num_inputs, Self.steps, Self.num_paths), MutAnyOrigin] 

    var result_buffer: DeviceBuffer[DType.float32]
    var grad_buffer:   DeviceBuffer[DType.float32]

    var result_tensor: Self.ResultTensorType
    var grad_tensor:   Self.GradTensorType

    var strike:   Float32
    var slippage: Float32

    fn __init__(out self, ctx: DeviceContext, strike: Float32, slippage: Float32) raises:
        self.ctx = ctx
        self.strike = strike
        self.slippage = slippage
        self.result_buffer = ctx.enqueue_create_buffer[DType.float32](1)
        self.grad_buffer   = ctx.enqueue_create_buffer[DType.float32](1 * Self.steps * Self.num_paths)
        self.result_tensor = Self.ResultTensorType(self.result_buffer)
        self.grad_tensor   = Self.GradTensorType(self.grad_buffer)

    fn value(self) raises -> Float32:
        host_buffer = self.ctx.enqueue_create_host_buffer[DType.float32](1)
        host_tensor = LayoutTensor[DType.float32, Layout.row_major(1), MutAnyOrigin](host_buffer)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.result_buffer)
        self.ctx.synchronize()

        return host_tensor[0][0]

    fn print_loss(self) raises:
        print()
        print('==== loss  ====')
        print_matrix_2[1, 1](self.ctx, self.result_buffer)

    fn print_grad(self) raises:
        print()
        print('==== loss grad ====')
        print_matrix_special[1, Self.steps, Self.num_paths](self.ctx, self.grad_buffer)

    fn apply[layout: Layout](
            self, 
            input_tensor: LayoutTensor[DType.float32, layout, MutAnyOrigin]
    ) raises:

        fn european_call_loss_kernel(
                input_tensor:  Self.InputTensorType,
                result_tensor: Self.ResultTensorType,
                strike:        Float32,
                slippage:      Float32,
        ):
            path = Int(thread_idx.x)
            value: Float32 = 0
            for step in range(1, Self.steps):
                value += input_tensor[2, step, path][0] * (
                            input_tensor[1,     step, path][0] * (1 + slippage) - 
                            input_tensor[1, step - 1, path][0] * (1 - slippage)
                        )

            payoff: Float32 = max(input_tensor[1, Self.steps - 1, path][0] - strike, 0)
            error:  Float32 = (value - payoff) ** 2

            total_error = block.sum[block_size=TPB, broadcast=False](val=SIMD[DType.float32, 1](error))

            cluster_arrive()
            cluster_wait()

            if thread_idx.x == 0:
                result_tensor[0] = total_error

        self.ctx.enqueue_function_experimental[european_call_loss_kernel](
                input_tensor,
                self.result_tensor,
                self.strike,
                self.slippage,
                grid_dim=(1),
                block_dim=(Self.num_paths),
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
                slippage:     Float32,
        ):
            path = Int(thread_idx.x)

            if path >= Self.num_paths:
                return

            value: Float32 = 0

            for step in range(Self.steps - 1):
                value += input_tensor[2, step + 1, path] [0] * (
                        input_tensor[1, step + 1, path][0] * (1 + slippage) -
                        input_tensor[1, step    , path][0] * (1 - slippage)
                    )

            payoff = max(input_tensor[1, Self.steps - 1, path] - strike, 0)
            grad_tensor[0, Self.steps, path] = 0

            for step in range(Self.steps - 1):
                grad_tensor[0, step, path] = 2 * (value - payoff) * (
                        input_tensor[1, step + 1, path] * (1 + slippage) -
                        input_tensor[1, step,     path] * (1 - slippage)
                    )

        self.ctx.enqueue_function_experimental[european_call_grad_kernel](
                input_tensor,
                self.grad_tensor,
                self.strike,
                self.slippage,
                grid_dim=(1),
                block_dim=(Self.num_paths),
        )
        self.ctx.synchronize()


fn generate_paths[num_inputs: Int, steps: Int, num_paths: Int](
        ctx: DeviceContext,
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutAnyOrigin],
        drift:        Float32,
        vol:          Float32,
        seed:         Int,
) raises:

    fn generate_paths_kernel(
            input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutAnyOrigin],
            drift:        Float32,
            vol:          Float32,
            seed:         Int,
    ):
        path = Int(thread_idx.x)

        if path >= num_paths or block_idx.x > 0:
            return

        input_tensor[0, 0, path] = 1
        input_tensor[1, 0, path] = 1
        input_tensor[2, 0, path] = 0

        dt = Float32(1.0 / (steps - 1))

        thread_seed = seed * Int(block_dim.x + thread_idx.x)
        random = NormalRandom(seed=thread_seed)

        for step in range(1, steps):
            if step < steps - 1:
                input_tensor[0, step, path] = 1 - dt * step
            else:
                input_tensor[0, step, path] = 0

            z = Float32(random.step_normal()[0])
            input_tensor[1, step, path] = input_tensor[1, step - 1, path] * exp((drift - 0.5 * vol ** 2)*dt + vol * sqrt(dt) * z)
            input_tensor[2, step, path] = 0

    ctx.enqueue_function_experimental[generate_paths_kernel](
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
        output_tensor: LayoutTensor[DType.float32, Layout.row_major(1,          steps, num_paths), MutAnyOrigin],
        input_tensor:  LayoutTensor[DType.float32, Layout.row_major(input_size, steps, num_paths), MutAnyOrigin],
        step: Int
) raises:

    fn next_step_kernel(
        output_tensor: LayoutTensor[DType.float32, Layout.row_major(1,          steps, num_paths), MutAnyOrigin],
        input_tensor:  LayoutTensor[DType.float32, Layout.row_major(input_size, steps, num_paths), MutAnyOrigin],
        step: Int
    ):
        path = Int(thread_idx.x)

        if path >= num_paths or block_idx.x > 0:
            return

        input_tensor[2, step + 1, path] = output_tensor[0, step, path]

    ctx.enqueue_function_experimental[next_step_kernel](
            output_tensor,
            input_tensor,
            step,
            grid_dim=(1),
            block_dim=(num_paths),
    )
    ctx.synchronize()


fn update_loss_grad[num_inputs: Int, steps: Int, num_paths: Int](
        ctx: DeviceContext,
        grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutAnyOrigin],
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutAnyOrigin],
        step:         Int,
) raises:

    fn update_loss_grad_kernel(
        grad_tensor:  LayoutTensor[DType.float32, Layout.row_major(1, steps, num_paths),          MutAnyOrigin],
        input_tensor: LayoutTensor[DType.float32, Layout.row_major(num_inputs, steps, num_paths), MutAnyOrigin],
        step:         Int,
    ):
        path = Int(thread_idx.x)
        step_idx = Int(block_idx.x)

        if path >= num_paths or step >= steps:
            return

        if step_idx <= step:
            grad_tensor[0, step_idx, path] = input_tensor[2, step_idx + 1, path]
        else:
            grad_tensor[0, step_idx, path] = 0.0

    ctx.enqueue_function_experimental[update_loss_grad_kernel](
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

    comptime inputs       = 3
    comptime steps        = 20
    comptime network_size = 16
    comptime num_paths    = 1024

    learning_rate: Float32 = 1e-3
    lrd_1:         Float32 = 0.95
    lrd_2:         Float32 = 10_000
    beta1:         Float32 = 0.9
    beta2:         Float32 = 0.999
    eps:           Float32 = 10e-8
    weight_decay:  Float32 = 0.01

    drift:  Float32   = 0
    vol:    Float32   = 0.2
    strike: Float32   = 1.1
    slippage: Float32 = 0.01

    layer_1 = DenseLayer[network_size, inputs,       steps, num_paths, ReluActivation](ctx, 'layer 1', learning_rate, lrd_1, lrd_2, beta1, beta2, eps, weight_decay)
    layer_2 = DenseLayer[network_size, network_size, steps, num_paths, ReluActivation](ctx, 'layer 2', learning_rate, lrd_1, lrd_2, beta1, beta2, eps, weight_decay)
    layer_3 = DenseLayer[1,            network_size, steps, num_paths, TanhActivation](ctx, 'layer 3', learning_rate, lrd_1, lrd_2, beta1, beta2, eps, weight_decay)
    loss    = EuropeanCallLoss[inputs, steps, num_paths](ctx, strike, slippage)

    pyray = Python.import_module('pyray')
    pyray.init_window(1260, 750, 'deep hedning')
    batch = 0

    test_path_buffer = ctx.enqueue_create_host_buffer[DType.float32](inputs * steps * num_paths)
    test_path_tensor = LayoutTensor[DType.float32, Layout.row_major(inputs, steps, num_paths), MutAnyOrigin](test_path_buffer)

    start_time = monotonic()

    while not pyray.window_should_close():
        pyray.begin_drawing()
        pyray.clear_background(pyray.BLACK)

        generate_paths[inputs, steps, num_paths](ctx, layer_1.in_tensor, drift, vol, batch)

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


        # Draw stuff
        if batch % 100 == 0:

            margin = 10

            # Text
            batch_message = 'batch: ' + String(batch)
            pyray.draw_text(batch_message, margin, margin, 20, pyray.GRAY)

            loss_message = 'loss: ' + String(loss.value())
            pyray.draw_text(loss_message, margin + 200, margin, 20, pyray.GRAY)

            elapsed_seconds = (monotonic() - start_time) // 1_000_000_000
            elapsed_message = 'time (seconds): ' + String(elapsed_seconds)
            pyray.draw_text(elapsed_message, margin + 400, margin, 20, pyray.GRAY)

            # Test paths
            test_paths = 9

            offset = 40
            plot_height = 220
            plot_width  = 400

            test_path_seed = 1
            generate_paths[inputs, steps, num_paths](ctx, layer_1.in_tensor, drift, vol, test_path_seed)

            for step in range(steps - 1):
                layer_1.apply(step)
                layer_1.feed_next(layer_2.in_tensor, step)
                layer_2.apply(step)
                layer_2.feed_next(layer_3.in_tensor, step)
                layer_3.apply(step)

                next_step[inputs, steps, num_paths](ctx, layer_3.out_tensor, layer_1.in_tensor, step)

            ctx.enqueue_copy(dst_buf=test_path_buffer, src_buf=layer_1.in_buffer)
            ctx.synchronize()

            for test_path in range(test_paths):
                y_offset = (test_path % 3)  * (plot_height + 2 * margin)
                x_offset = (test_path // 3) * (plot_width + 2 * margin)

                # Vertical
                pyray.draw_line(
                        margin + x_offset, 
                        offset + y_offset,
                        margin + x_offset, 
                        offset + y_offset + plot_height, 
                        pyray.GRAY
                )
                pyray.draw_line(
                        margin + x_offset +plot_width, 
                        offset + y_offset, 
                        margin + x_offset + plot_width, 
                        offset + y_offset + plot_height, 
                        pyray.GRAY
                )

                # Horizontal
                pyray.draw_line(
                        margin + x_offset, 
                        offset + y_offset, 
                        margin + x_offset + plot_width, 
                        offset + y_offset, 
                        pyray.GRAY
                )
                pyray.draw_line(
                        margin + x_offset, 
                        offset + y_offset + plot_height // 2, 
                        margin + x_offset + plot_width, 
                        offset + y_offset + plot_height // 2, 
                        pyray.GRAY
                )
                pyray.draw_line(
                        margin + x_offset, 
                        offset + y_offset + plot_height, 
                        margin + x_offset + plot_width, 
                        offset + y_offset + plot_height, 
                        pyray.GRAY
                )

                y_mid:   Int     = offset + y_offset + plot_height // 2
                y_scale: Float32 = Float32(plot_height / 2)
                dx:      Float32 = Float32(plot_width / (steps - 2))

                # Strike
                pyray.draw_line(
                        margin + x_offset, 
                        y_mid - Int((strike - 1) * y_scale), 
                        margin + x_offset + plot_width, 
                        y_mid - Int((strike - 1) * y_scale), 
                        pyray.RED,
                )

                for step in range(steps - 2):

                    # Hedge
                    pyray.draw_line(
                            margin + x_offset + Int(dx * step), 
                            y_mid - Int(test_path_tensor[2, step + 1, test_path][0] * y_scale), 
                            margin + x_offset + Int(dx * (step + 1)), 
                            y_mid - Int(test_path_tensor[2, step + 2, test_path][0] * y_scale), 
                            pyray.ORANGE,
                    )

                    #Stock
                    pyray.draw_line(
                            margin + x_offset + Int(dx * step), 
                            y_mid - Int((test_path_tensor[1, step, test_path][0] - 1) * y_scale), 
                            margin + x_offset + Int(dx * (step + 1)), 
                            y_mid - Int((test_path_tensor[1, step + 1, test_path][0] - 1) * y_scale), 
                            pyray.BLUE,
                    )

            pyray.end_drawing()

        batch += 1

    pyray.close_window()
