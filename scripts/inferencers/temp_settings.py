from contextlib import contextmanager


@contextmanager
def temp_attr(p, **kwargs):
    backup = {}
    for key, new_value in kwargs.items():
        if hasattr(p, key):
            backup[key] = getattr(p, key)
            setattr(p, key, new_value)
    try:
        yield
    finally:
        for key, original_value in backup.items():
            if hasattr(p, key):
                setattr(p, key, original_value)


@contextmanager
def temp_dict(dict_obj, **kwargs):
    backup = {}
    for key, new_value in kwargs.items():
        if key in dict_obj:
            backup[key] = dict_obj.get(key)
            dict_obj[key] = new_value
    try:
        yield
    finally:
        for key, original_value in backup.items():
            if key in dict_obj:
                dict_obj[key] = original_value
