DATA_REGISTRY = dict()


def register_dataset(cls):
    DATA_REGISTRY[cls.__name__] = cls
    return cls


from data import esc_dataset, urbansound8k, gtzan


__all__ = ["register_dataset"]