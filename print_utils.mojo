from math import floor, log10
from layout.layout_tensor import Layout, LayoutTensor
from gpu.host import DeviceContext, DeviceBuffer

fn print_matrix_2[M: Int, N: Int](ctx: DeviceContext, device_buffer: DeviceBuffer[DType.float32]) raises -> None:
    host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    host_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N), MutableAnyOrigin](host_buffer)

    ctx.enqueue_copy(dst_buf=host_buffer, src_buf=device_buffer)
    ctx.synchronize()

    for i in range(M):
        var row = ''
        for j in range(N):
            row += padded_f32(host_tensor[i, j][0]) + '  '
        print(row)


fn print_matrix_3[M: Int, N: Int, K: Int](ctx: DeviceContext, device_buffer: DeviceBuffer[DType.float32], k: Int) raises -> None:
    host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M * N * K)
    host_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N, K), MutableAnyOrigin](host_buffer)

    ctx.enqueue_copy(dst_buf=host_buffer, src_buf=device_buffer)
    ctx.synchronize()

    for i in range(M):
        var row = ''
        for j in range(N):
            row += padded_f32(host_tensor[i, j, k][0]) + '  '
        print(row)

fn print_matrix_special[M: Int, N: Int, K: Int](ctx: DeviceContext, device_buffer: DeviceBuffer[DType.float32]) raises -> None:
    host_buffer = ctx.enqueue_create_host_buffer[DType.float32](M * N * K)
    host_tensor = LayoutTensor[DType.float32, Layout.row_major(M, N, K), MutableAnyOrigin](host_buffer)

    ctx.enqueue_copy(dst_buf=host_buffer, src_buf=device_buffer)
    ctx.synchronize()

    for i in range(K):
        var row = ''
        for j in range(N):
            row += padded_f32(host_tensor[0, j, i][0]) + '  '
        print(row)

fn padded_f32(x: Float32) -> String:
    """
    Horrible stuff.
    """
    var chars: Int
    if x < 0:
        chars = 9
    else:
        chars = 8

    var e = Int(floor(log10(abs(x))))
    var res: String


    if e >= chars - 3:
        var s = String(x)
        var w = '' + s[0] if x > 0 else '-' + s[1]
        if e < 10:
            res = w + '.' + String(s[1:5]) + 'e' + String(e)
        elif e < 100:
            res = w + '.' + String(s[1:4]) + 'e' + String(e)
        elif e < 1000:
            res = w + '.' + String(s[1:3]) + 'e' + String(e)
        else:
            return '       inf'
    else:
        var s = String(x)
        var parts = s.split('.')
        res = parts[0] + '.'
        var decimals = chars - len(parts[0])
        for i in range(decimals - 1):
            if i > len(parts[1]) - 1:
                res += '0'
            else:
                res += parts[1][i]

    if x >= 0:
        return ' ' + res
    else:
        return res

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
