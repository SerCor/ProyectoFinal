from .compose import Value
from functools import wraps


def with_value(f):
    '''Decorador para encapsular el retorno de una funcion como
    un objeto de tipo Value'''
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Value(f(*args, **kwargs))

    return wrapper


def with_log(f):
    '''Decorator para imprimir el inicio y final de una función'''
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f'Inicio de función {f.__name__}')
        r = f(*args, **kwargs)
        print(f'Fin de función {f.__name__}')
        
        return r

    return wrapper
