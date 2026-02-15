from math import floor, log10
from layout.layout_tensor import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer

fn print_matrix_2(
        ctx: DeviceContext, 
        M: Int, 
        N: Int, 
        device_buffer: DeviceBuffer[DType.float32]
) raises:
    host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    buffer = NDBuffer[dtype, 2, MutAnyOrigin](ptr=host_buffer.unsafe_ptr(), dynamic_shape=IndexList[2](M, N))

    ctx.enqueue_copy(dst_buf=host_buffer, src_buf=device_buffer)
    ctx.synchronize()

    for i in range(M):
        var row = ''
        for j in range(N):
            row += padded_f32(buffer[i, j][0]) + '  '
        print(row)

fn padded_f32(x: Float32) -> String:
    return String(x).ascii_rjust(12)

fn main():
    print(padded_f32(1))
    print(padded_f32(-1))
    print(padded_f32(12345))
    print(padded_f32(-10))
    print(padded_f32(19.9))
    print(padded_f32(-0.0343))
    print(padded_f32(0.343))
    print(padded_f32(3.43))
    print()
    print(padded_f32(123456))
    print(padded_f32(-1234567))
    print(padded_f32(12345678))
    print(padded_f32(-123456789))
    print(padded_f32(123456e10))
    print()
    print(padded_f32(0.0000000123))
    print(padded_f32(-0.000000123))
    print(padded_f32(0.00000123))
    print(padded_f32(-0.0000123))
    print(padded_f32(0.000123))
    print(padded_f32(-0.00123))
