# Bug Fixes from Web Rewrite

## he_init_kernel grid/block dimensions (FIXED)
The `he_init_kernel` had `grid_dim` and `block_dim` swapped. Since the kernel uses `thread_idx.x` and `thread_idx.y` to index into the weight matrix, the dimensions need to be in `block_dim`, not `grid_dim`.

**Wrong:**
```mojo
grid_dim=(self.M, self.N),
block_dim=(1),
```

**Correct:**
```mojo
grid_dim=(1),
block_dim=(self.M, self.N),
```

## Other fixes applied during debugging:
1. `european_call.mojo` bwd_kernel: Fixed y buffer shape from `(inputs, ...)` to `(1, ...)`
2. `dense_layer.mojo` downstream_kernel: Fixed d buffer shape from `(M, ...)` to `(N, ...)`
3. `european_call.mojo` update_loss_grad: Changed from `x_buffer` to `d_buffer`
4. `dense_layer.mojo` he_init_kernel: Added unique seed per thread (`thread_seed = seed + UInt64(i * N + j)`)
