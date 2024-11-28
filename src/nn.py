from abc import ABC, abstractmethod


class Module(ABC):
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self): ...

    @abstractmethod
    def __call__(self): ...
