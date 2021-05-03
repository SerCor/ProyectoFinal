

def replace_list(line, replace_list, replace):
    '''Eliminar cualquier caracter contenido en la lista'''
    for replace_item in replace_list:
        line = line.replace(replace_item, replace)
    
    return line


def drop_list(line, drop_list):
    return replace_list(line, drop_list, '')


def drop_underscore(line):
    new_line = ''
    words = line.split()
    
    for w in words:
        if w.startswith('$'):
            new_line += ' ' + w
        else:
            new_line += ' ' + w.replace('_', '')
    
    return new_line +'\n'


def load_stop_words():
    with open('text_processing/stopwords.txt', 'r') as file:
        return set(word.strip() for line in file for word in line.split())