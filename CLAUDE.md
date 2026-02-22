# Cher

Deep hedging implementation in Mojo GPU kernels, based on [Buehler et al. 2018](https://arxiv.org/abs/1802.03042).

## What deep hedging is

The network learns an optimal hedging strategy for derivatives (currently a European call option) by minimising the squared difference between the hedging portfolio P&L and the option payoff at expiry. The loss can only be determined after the final step, but every hedge decision along the path affects both the final P&L and cumulative transaction costs.

## Network architecture

3-layer feedforward network (3 -> 16 -> 16 -> 1) with ReLU, ReLU, Tanh activations. Tanh bounds the output (hedge ratio / delta) to [-1, 1]. The same weights are shared across all timesteps — applied recurrently through time.

### Inputs per timestep (3 channels)

- Channel 0: Time remaining (linearly from 1 to 0)
- Channel 1: Stock price (GBM with drift/vol, starting at 1)
- Channel 2: The network's own delta output from the previous step (0 at t=0)

### Why the recurrence (channel 2) matters

Without transaction costs, the optimal hedge at each step depends only on current market state (time, stock price). Each step would be independent — no need for the network to know its previous output.

With transaction costs, the cost of rebalancing depends on how much the position changes. The optimal hedge at step t depends on what the network is currently holding (delta_{t-1}). The network needs its previous output as input to decide whether the improvement from rebalancing justifies the transaction cost. This is what creates the recurrence.

### Loss function

For each Monte Carlo path:
- P&L = sum over steps of: delta * (S_next * (1 + slippage) - S_prev * (1 - slippage))
- Payoff = max(S_T - strike, 0)
- Loss = mean over paths of (P&L - payoff)^2

The slippage term models proportional transaction costs on both sides of each trade.

## Forward pass

For each timestep:
1. Layer 1 forward (input -> 16, ReLU)
2. Copy layer 1 output to layer 2 input
3. Layer 2 forward (16 -> 16, ReLU)
4. Copy layer 2 output to layer 3 input
5. Layer 3 forward (16 -> 1, Tanh) — produces delta
6. `next_step`: copy delta into input channel 2 of the next timestep (the recurrent connection)

After all steps, compute the loss.

## Backward pass (BPTT)

This is the main novel implementation detail.

1. `loss.bwd()` computes the direct dL/d(delta) at each step — treating each delta as if it only affects the loss through its own hedging term.

2. Then iterating backwards from the last step:
   - The gradient in `loss.grad_buffer` is backpropped through layers 3 -> 2 -> 1
   - Each layer's `bwd` computes downstream gradients AND performs an Adam weight update
   - After layer 1's backward, `layer_1.d_buffer[2, step+1, ...]` contains the gradient flowing through the recurrent connection — how the loss changes w.r.t. delta at this step via the chain: delta_t -> input channel 2 at step t+1 -> delta_{t+1} -> ...
   - `update_loss_grad(step)` extracts this recurrent gradient and sets it as the upstream gradient for the next backward iteration, zeroing out already-processed steps

3. Each backward iteration does its own Adam step. So rather than accumulating all BPTT gradients into one update, each depth of the recurrence chain contributes a separate update.

## Optimizer

AdamW (Adam with decoupled weight decay). Learning rate decays as: lr * lr_d1^(counter / lr_d2).

## File structure

- `european_call.mojo` — GPU Network, EuropeanCallLoss, Params, path generation, forward/backward pass, BPTT loop
- `dense_layer.mojo` — GPU DenseLayer with forward, backward (downstream gradients + Adam update), GPU kernels
- `activation.mojo` — ReluActivation, TanhActivation (shared by both backends)
- `cher_mojo.mojo` — Python bindings (PythonNetwork, TestPath) wrapping the Mojo Network for use from worker.py
- `worker.py` — Subprocess that owns the Network, communicates with app.py over a Unix socket, runs the training loop
- `app.py` — FastAPI web server, bridges WebSocket clients to worker processes
- `static/index.html` — Frontend with Plotly chart, parameter controls, live loss and step time display

## Web architecture

Browser <-> WebSocket <-> app.py (FastAPI) <-> Unix socket <-> worker.py (owns Network)

Each WebSocket connection spawns a dedicated worker subprocess. Messages between app.py and worker.py use a binary protocol (struct-packed). The worker runs `network.run()` in a non-blocking loop, sending loss and test path updates every BATCH_UPDATE_SIZE iterations.
