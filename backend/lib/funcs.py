"""
Function lib
"""
from typing import Any, Tuple, List, Dict, Any, Callable, Optional
import json

def group_by(fs: List[Callable[[Any], Any]], ls: List[Any]) -> Dict[Any, Any]:
    """
    Recursivly divides a list into sublists according to a list of membership-defining functions
    """
    match fs:
        case [f]:
            match ls:
                case []: return {}
                case _:
                    d = {}
                    for x in ls:
                        k = f(x)
                        if k in d:
                            d[k].append(x)
                        else:
                            d[k] = [x]
                    return d
        case [f, *fs]:
            d = group_by([f], ls)
            return {k: group_by(fs, v) for k, v in d.items()}


def pretty(x):
    """
    A convenience pretty printing function
    """
    print( json.dumps(x, indent=4) )



