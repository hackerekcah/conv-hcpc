ARCH_REGISTRY = dict()


def register_arch(cls):
    ARCH_REGISTRY[cls.__name__] = cls
    return cls


# explicitly import all models to register arch
from models import conv1d


__all__ = ["register_arch"]