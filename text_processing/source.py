from .decorators import with_value


@with_value
def from_file(path):
    '''Crea un pipeline que tiene como fuente
    un archivo'''
    with open(path, encoding='utf-8', mode='r') as file:
        for line in file:
            yield line


@with_value
def from_fd(fd):
    '''Crea un pipeline que tiene como fuente
    un descritor de archivo'''
    return fd