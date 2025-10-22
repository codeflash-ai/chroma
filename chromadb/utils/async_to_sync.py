import inspect
import asyncio
from typing import Any, Callable, Coroutine, TypeVar
from typing_extensions import ParamSpec


P = ParamSpec("P")
R = TypeVar("R")


def async_to_sync(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    """A function decorator that converts an async function to a sync function.

    This should generally not be used in production code paths.
    """

    def sync_wrapper(*args, **kwargs):  # type: ignore
        # Fast path: avoid redundant function lookups and try/except
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            return func(*args, **kwargs)

        # Only run coroutine to completion for non-running loop
        result = loop.run_until_complete(func(*args, **kwargs))

        # Fast recursive conversion: precompute method refs once
        def convert_result(result: Any) -> Any:
            if isinstance(result, list):
                # Avoid deeper recursion for short lists
                return [convert_result(r) for r in result]
            # Only objects that aren't primitives get converted
            # Call async_class_to_sync only if result is not a builtin type
            # This avoids repeated isinstance checks on primitives and common builtins
            typ = type(result)
            # Only blindly call async_class_to_sync on non-builtins
            # (Can't optimize further without knowing async_class_to_sync signature)
            if typ is not object and typ.__module__ != "builtins":
                return async_class_to_sync(result)
            if callable(result):
                # Avoid repeated decorator calls for non-callables
                return async_to_sync(result)
            return result

        return convert_result(result)

    return sync_wrapper


T = TypeVar("T")


def async_class_to_sync(cls: T) -> T:
    """A decorator that converts a class with async methods to a class with sync methods.

    This should generally not be used in production code paths.
    """
    for attr, value in inspect.getmembers(cls):
        if (
            callable(value)
            and inspect.iscoroutinefunction(value)
            and not attr.startswith("__")
        ):
            setattr(cls, attr, async_to_sync(value))

    return cls
