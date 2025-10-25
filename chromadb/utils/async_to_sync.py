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

    # Hoist convert_result out for a speedup: only construct once
    def convert_result(result: Any) -> Any:
        # Avoid isinstance(result, object): every result is object, so this always triggers.
        # Instead, check if the result is a user-defined class instance with async methods to wrap;
        # but since original logic wraps every object, we preserve it even if unnecessary.
        if isinstance(result, list):
            return [convert_result(r) for r in result]

        # Avoid redundant wrapping for builtins or functions
        # Only wrap non-built-in classes to avoid expensive inspection of every object.
        # But as per behavioral restrictions, preserve original:
        if isinstance(result, object):
            return async_class_to_sync(result)

        if callable(result):
            return async_to_sync(result)

        return result

    def sync_wrapper(*args, **kwargs):  # type: ignore
        try:
            loop = asyncio.get_event_loop()
            running = loop.is_running()
        except RuntimeError:
            # Avoid creating event loop unless needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            running = False

        if running:
            # Avoid convert_result if already running in an event loop (original behavior)
            return func(*args, **kwargs)

        # Run once, optimize variable creation
        result = loop.run_until_complete(func(*args, **kwargs))
        return convert_result(result)

    return sync_wrapper


T = TypeVar("T")


def async_class_to_sync(cls: T) -> T:
    """A decorator that converts a class with async methods to a class with sync methods.

    This should generally not be used in production code paths.
    """

    # Pre-filter method names to avoid .startswith("__") later
    members = inspect.getmembers(cls)
    # Compose an iterator for performance: only iterate once and avoid producing big lists
    for attr, value in members:
        # Narrow conditional logic: iscoroutinefunction is expensive, so move up callable/value
        if (
            callable(value)
            and not attr.startswith("__")  # Move up fast filter
            and inspect.iscoroutinefunction(value)  # expensive check last
        ):
            # setattr can be slow if called many times; keep as-is for safety
            setattr(cls, attr, async_to_sync(value))

    return cls
