from math import tanh


trait Activation:

    @staticmethod
    fn apply(y: Float32) -> Float32:
        ...

    @staticmethod
    fn apply_grad(y: Float32) -> Float32:
        ...


struct ReluActivation(Activation):

    @staticmethod
    fn apply(y: Float32) -> Float32:
        return max(y, 0)

    @staticmethod
    fn apply_grad(y: Float32) -> Float32:
        if y > 0:
            return 1
        else:
            return 0


struct TanhActivation(Activation):

    @staticmethod
    fn apply(y: Float32) -> Float32:
        return tanh(y)

    @staticmethod
    fn apply_grad(y: Float32) -> Float32:
        return 1 - y ** 2

