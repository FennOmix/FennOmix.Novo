"""Public Python API for FoxNovo."""


def train(*args, **kwargs):
    from .api_denovo import train as train_impl

    return train_impl(*args, **kwargs)


def predict(*args, **kwargs):
    from .api_denovo import predict as predict_impl

    return predict_impl(*args, **kwargs)


__all__ = ["train", "predict"]
