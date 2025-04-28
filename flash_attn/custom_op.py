import inspect
from typing import Callable, Iterable, Sequence

import torch


def _dispatch(func: Callable, compileable_fn: Callable, *args, **kwargs):
    if torch.compiler.is_compiling():
        output = compileable_fn(*args, **kwargs)
    else:
        output = func(*args, **kwargs)

    return output


def torch_custom_op(
    name: str = None,
    mutates_args: str | Iterable[str] = None,
    device_types: str | Sequence[str] | None = None,
    schema: str | None = None,
    fake_func: Callable | None = None,
) -> Callable:
    def _inner(func: Callable):
        if torch.__version__ >= "2.4.0":
            compileable_func = torch.library.custom_op(
                name, func, mutates_args=mutates_args, device_types=device_types, schema=schema
            )

            if fake_func is not None:
                compileable_func.register_fake(fake_func)
        else:
            compileable_func = func

        def _run(*args, **kwargs):
            return _dispatch(func, compileable_func, *args, **kwargs)

        _run.__signature__ = inspect.signature(func)
        _run.__name__ = func.__name__
        _run.__doc__ = func.__doc__

        return _run

    return _inner
