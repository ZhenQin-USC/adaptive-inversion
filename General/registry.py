MULTIFIELD_LOSS_REGISTRY = {}
SINGLEFIELD_LOSS_REGISTRY = {}


def register_singlefield_loss(name):
    def decorator(cls):
        SINGLEFIELD_LOSS_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def register_multifield_loss(name):
    def decorator(cls):
        MULTIFIELD_LOSS_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_multifield_loss(name: str, **kwargs):
    name = name.lower()
    if name not in MULTIFIELD_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: '{name}'")
    return MULTIFIELD_LOSS_REGISTRY[name](**kwargs)

def get_singlefield_loss(name: str, **kwargs):
    name = name.lower()
    if name not in SINGLEFIELD_LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: '{name}'")
    return SINGLEFIELD_LOSS_REGISTRY[name](**kwargs)
