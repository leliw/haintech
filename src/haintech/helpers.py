from typing import Any, List, get_args, get_origin


def is_list_type(t: Any) -> bool:
    """Check if a type is a list type.

    Args:
        t: The type to check.
    Returns:
        True if the type is a list type, False otherwise.
    """
    origin = get_origin(t)
    return origin is list or origin is List


def get_inner_type(t: Any) -> Any:
    """Get the inner type of a list type.

    Args:
        t: The list type.
    Returns:
        The inner type of the list type.
    """
    if not is_list_type(t):
        return None
    args = get_args(t)
    return args[0] if args else Any
