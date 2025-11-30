based on:
https://arxiv.org/abs/1802.03042

- deep hedging implemented in mojo GPU kernels
- recurrent, but didn't add transaction costs yet
- use python raylib to monitor training (pip install raylib)
- the nn is completely independent of the model
    - can use any way to generate paths
    - can define any payoff (including path dependent)

European call example
- orange line: hedge position (between -1 & 1)
- blue line:   stock price for this path
- red line:    option strike

![Description](screen_1.png)
![Description](screen_2.png)
