from sys import has_accelerator
from layout.layout_tensor import Layout, LayoutTensor
from matmul import he_init, matmul, vector_add, relu, tanh, gbm_paths
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute, HostBuffer


fn get_ctx() raises -> DeviceContext:

    @parameter
    if not has_accelerator():
        raise Error('No compatible GPU found')
    else:
        ctx = DeviceContext()
        print('Running on', ctx.name())
        print('MULTIPROCESSOR_COUNT                ', ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT))
        print('MAX_BLOCKS_PER_MULTIPROCESSOR       ', ctx.get_attribute(DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR))
        print('MAX_THREADS_PER_BLOCK               ', ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK))
        print('WARP_SIZE                           ', ctx.get_attribute(DeviceAttribute.WARP_SIZE))
        print('MAX_SHARED_MEMORY_PER_BLOCK         ', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK))
        print('MAX_SHARED_MEMORY_PER_MULTIPROCESSOR', ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR))
        print()

    return ctx


struct GBMPaths[M: Int, N: Int]:

    alias layout: Layout = Layout.row_major(M, N)

    alias random_seed: Int = 43
    alias random_size: Int = 4  # Number of random variables generated on GPU in one call
    alias random_draws = (N * M + Self.random_size - 1) // Self.random_size

    var ctx:         DeviceContext
    var buffer:      DeviceBuffer[DType.float32]
    var host_buffer: HostBuffer[DType.float32]
    var tensor:      LayoutTensor[DType.float32, Self.layout, MutableAnyOrigin]
    var host_tensor: LayoutTensor[DType.float32, Self.layout, MutableAnyOrigin]


    fn __init__(out self, ctx: DeviceContext, mu: Float32, sigma: Float32, dt: Float32) raises:
        self.ctx = ctx
        self.buffer = ctx.enqueue_create_buffer[DType.float32](M * N)
        self.tensor = LayoutTensor[DType.float32, self.layout, MutableAnyOrigin](self.buffer)
        self.host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M)
        self.host_tensor = LayoutTensor[DType.float32, self.layout, MutableAnyOrigin](self.host_buffer)

        ctx.enqueue_function[gbm_paths[self.layout]
        ](
            self.tensor,
            mu, 
            sigma, 
            dt,
            self.random_seed,
            grid_dim=(1),
            block_dim=(self.random_draws),
        )

    fn print_values(self) raises:
        host_buffer = self.ctx.enqueue_create_host_buffer[DType.float32](M * N)
        host_tensor = LayoutTensor[DType.float32, self.layout, MutableAnyOrigin](host_buffer)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.buffer)
        self.ctx.synchronize()

        print()
        print('==== paths ====')
        for i in range(M):
            var row = ''
            for j in range(N):
                val = host_tensor[i, j]
                if val < 0:
                    row += String(val)[:7] + '  '
                else:
                    row += ' ' + String(val)[:6] + '  '
            print(row)

struct InputLayer[M: Int]:

    alias layout: Layout = Layout.row_major(M, 1)

    var ctx:         DeviceContext
    var buffer:      DeviceBuffer[DType.float32]
    var host_buffer: HostBuffer[DType.float32]
    var tensor:      LayoutTensor[DType.float32, Self.layout, MutableAnyOrigin]
    var host_tensor: LayoutTensor[DType.float32, Self.layout, MutableAnyOrigin]

    fn __init__(out self, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.buffer = ctx.enqueue_create_buffer[DType.float32](M)
        self.tensor = LayoutTensor[DType.float32, self.layout, MutableAnyOrigin](self.buffer)
        self.host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M)
        self.host_tensor = LayoutTensor[DType.float32, self.layout, MutableAnyOrigin](self.host_buffer)

    fn __setitem__(self, idx: Int, value: Float32):
        self.tensor[idx, 0] = Scalar[DType.float32](value)

    fn sync_to_gpu(self) raises:
        self.ctx.enqueue_copy(dst_buf=self.buffer, src_buf=self.host_buffer)
        self.ctx.synchronize()

    fn print_values(self) raises:
        print()
        print('==== input ====')
        for i in range(M):
            val = self.host_tensor[i, 0]
            if val < 0:
                print(String(val)[:7])
            else:
                print('', String(val)[:6])


struct DenseLayer[
        M: Int, 
        N: Int,
        activation_function: fn(tensor: LayoutTensor[DType.float32, Layout.row_major(M, 1), MutableAnyOrigin]),
]:

    alias weight_layout: Layout = Layout.row_major(M, N)
    alias in_layout:     Layout = Layout.row_major(N, 1)
    alias out_layout:    Layout = Layout.row_major(M, 1)

    alias random_seed: Int = 43
    alias random_size: Int = 4  # Number of random variables generated on GPU in one call
    alias random_draws = (N * M + Self.random_size - 1) // Self.random_size

    var layer_name: String
    var ctx: DeviceContext

    var weight_buffer: DeviceBuffer[DType.float32]
    var bias_buffer:   DeviceBuffer[DType.float32]
    var out_buffer:    DeviceBuffer[DType.float32]

    var weight_tensor: LayoutTensor[DType.float32, Self.weight_layout, MutableAnyOrigin]
    var bias_tensor:   LayoutTensor[DType.float32, Self.out_layout,    MutableAnyOrigin]
    var out_tensor:    LayoutTensor[DType.float32, Self.out_layout,    MutableAnyOrigin]

    fn __init__(out self, ctx: DeviceContext, layer_name: String) raises:
        self.ctx = ctx
        self.layer_name = layer_name

        self.weight_buffer = ctx.enqueue_create_buffer[DType.float32](M * N)
        self.bias_buffer   = ctx.enqueue_create_buffer[DType.float32](M)
        self.out_buffer    = ctx.enqueue_create_buffer[DType.float32](M)

        self.weight_tensor = LayoutTensor[DType.float32, self.weight_layout, MutableAnyOrigin](self.weight_buffer)
        self.bias_tensor   = LayoutTensor[DType.float32, self.out_layout,    MutableAnyOrigin](self.bias_buffer)
        self.out_tensor    = LayoutTensor[DType.float32, self.out_layout,    MutableAnyOrigin](self.out_buffer)

        if M * N > ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK):
            raise Error('Not implemented')

        ctx.enqueue_function[he_init[self.weight_layout]
        ](
            self.weight_tensor,
            self.random_seed,
            grid_dim=(1),
            block_dim=(self.random_draws),
        )

    fn apply(self, tensor: LayoutTensor) raises:
        if tensor.dim(0) != N or tensor.dim(1) != 1:
            raise Error('Invalid shape:', tensor.dim(0), 'x', tensor.dim(1))

        self.ctx.enqueue_function[matmul[self.weight_layout, self.in_layout, self.out_layout]
        ](
            self.weight_tensor,
            tensor,
            self.out_tensor,
            grid_dim=(1),
            block_dim=(M * N), # Todo > 1024
        )

        self.ctx.synchronize()

        self.ctx.enqueue_function[vector_add[self.out_layout]
        ](
            self.out_tensor,
            self.bias_tensor,
            grid_dim=(1),
            block_dim=(M),
        )

        self.ctx.synchronize()

        self.ctx.enqueue_function[activation_function
        ](
            self.out_tensor,
            grid_dim=(1),
            block_dim=(M),
        )

        self.ctx.synchronize()

    fn print_weights(self) raises:
        host_buffer = self.ctx.enqueue_create_host_buffer[DType.float32](M * N)
        host_tensor = LayoutTensor[DType.float32, self.weight_layout, MutableAnyOrigin](host_buffer)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.weight_buffer)
        self.ctx.synchronize()

        print()
        print('====', self.layer_name, 'weights ====')
        for i in range(M):
            var row = ''
            for j in range(N):
                val = host_tensor[i, j]
                if val < 0:
                    row += String(val)[:7] + '  '
                else:
                    row += ' ' + String(val)[:6] + '  '
            print(row)

    fn print_bias(self) raises:
        host_buffer = self.ctx.enqueue_create_host_buffer[DType.float32](M)
        host_tensor = LayoutTensor[DType.float32, self.out_layout, MutableAnyOrigin](host_buffer)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.bias_buffer)
        self.ctx.synchronize()

        print()
        print('====', self.layer_name, 'bias ====')
        for i in range(M):
            val = host_tensor[i, 0]
            if val < 0:
                print(String(val)[:7])
            else:
                print('', String(val)[:6])

    fn print_out(self) raises:
        host_buffer = self.ctx.enqueue_create_host_buffer[DType.float32](M)
        host_tensor = LayoutTensor[DType.float32, self.out_layout, MutableAnyOrigin](host_buffer)

        self.ctx.enqueue_copy(dst_buf=host_buffer, src_buf=self.out_buffer)
        self.ctx.synchronize()

        print()
        print('====', self.layer_name, 'out ====')
        for i in range(M):
            val = host_tensor[i, 0]
            if val < 0:
                print(String(val)[:7])
            else:
                print('', String(val)[:6])


fn main() raises:
    ctx = get_ctx()

    alias relu_16 = relu[Layout.row_major(16, 1)]
    alias tanh_1  = tanh[Layout.row_major(1 , 1)]
    alias steps = 25

    layer1 = DenseLayer[16, 3 , relu_16](ctx, 'layer 1')
    layer2 = DenseLayer[16, 16, relu_16](ctx, 'layer 2')
    layer3 = DenseLayer[1,  16, tanh_1 ](ctx, 'layer 3')

    paths = GBMPaths[10, steps](ctx, 0.1, 0.05, Float32(steps) / 365)
    paths.print_values()

    input_layer = InputLayer[3](ctx)
    input_layer.host_tensor[0, 0] = 1                     # Stock price
    input_layer.host_tensor[1, 0] = 0                     # Initial hedge
    input_layer.host_tensor[2, 0] = Float32(steps) / 365  # Time to maturity
    input_layer.sync_to_gpu()
    input_layer.print_values()

    for i in range(steps):
        layer1.apply(input_layer.tensor)
        layer2.apply(layer1.out_tensor)
        layer3.apply(layer2.out_tensor)
        layer3.print_out()


    # layer1.print_weights()
    # layer1.print_bias()
    # layer1.print_out()

    # layer2.print_weights()
    # layer2.print_bias()
    # layer2.print_out()

    # layer3.print_weights()
    # layer3.print_bias()
    # layer3.print_out()

    print('yeah')

