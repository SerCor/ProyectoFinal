class Value:
    '''Valor dentro de un pipeline'''
    def __init__(self, value):
        self.value = value

    def __rshift__(self, other):
        return other(self.value)