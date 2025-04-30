# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/_library/triton.py
# The PyTorch implementation simply ignores the schema argument, we simply modify it to use schema.

from typing import Optional, Callable, Iterable, Union

from torch.library import custom_op, CustomOpDef
from torch._library.triton import set_wrap_triton_enabled


def triton_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    schema: Optional[str] = None,
    # If allow_decomposition=True, this matches torch.library.triton_op behavior. If set to False,
    # then it behaves like torch.library.custom_op instead, which doesn't decompose the operator
    # and so inductor can't trace inside.
    allow_decomposition=True,
) -> Callable:
    def dec(fn: Callable[..., object]) -> CustomOpDef:
        def backend_fn(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Optimization: we're passing regular Tensors into the triton kernel, so
            # no need to go through HOP dispatch
            with set_wrap_triton_enabled(False):
                return fn(*args, **kwargs)

        result = custom_op(
            name,
            backend_fn,
            mutates_args=mutates_args,
            # This is the only difference with the PyTorch implementation
            schema=schema,
        )
        from torch._subclasses.functional_tensor import FunctionalTensorMode

        # We require that the user pass us a function that is make_fx traceable,
        # so we can just register it as the Fake/meta kernel.
        result.register_fake(fn)

        if allow_decomposition:
            # We decompose the operator when FunctionalTensorMode is active.
            # The goal is to decompose the operator in AOTDispatcher.
            # - With torch.compile, this means that the backend (usually Inductor)
            #   can see a call to the triton kernel(s) and so it can directly optimize
            #   them by inlining them into the lowering process.
            def functional_decomp(  # type: ignore[no-untyped-def]
                mode, op, types, args, kwargs
            ):
                from torch.export._trace import custom_triton_ops_decomposition_disabled

                if custom_triton_ops_decomposition_disabled():
                    return mode.__torch_dispatch__(op, types, args, kwargs)
                else:
                    with mode:
                        return fn(*args, **kwargs)

            result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)

        return result

    if fn is None:
        return dec
    else:
        return dec(fn)
