#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Callable, Dict, List, Tuple


class Arguments:
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, that: object) -> bool:
        if not isinstance(that, Arguments):
            return False
        if len(self.args) != len(that.args):
            return False
        for self_arg, that_arg in zip(self.args, that.args):
            if self_arg is not that_arg:
                return False
        self_keys = set(self.kwargs.keys())
        that_keys = set(that.kwargs.keys())
        if self_keys != that_keys:
            return False
        for key in self_keys:
            if self.kwargs[key] is not that.kwargs[key]:
                return False
        return True


def linear_cache_non_hashable(max_size: int) -> Callable:
    def decorator(function: Callable):
        cache: List[Tuple[Arguments, Any]] = []

        def wrapper(*args: List[Any], **kwargs: Dict[str, Any]) -> Any:
            arguments = Arguments(*args, **kwargs)
            for cached_arguments, cached_result in cache:
                if arguments == cached_arguments:
                    return cached_result
            result = function(*args, **kwargs)
            if len(cache) >= max_size:
                cache.pop(0)
            cache.append((arguments, result))
            return result

        return wrapper

    return decorator
