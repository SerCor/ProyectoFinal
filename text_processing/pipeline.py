import re
from .decorators import with_value, with_log
import string
from string import digits
from .tools import replace_list, drop_list, drop_underscore, load_stop_words
from .compose import Value
from nltk.stem.snowball import SnowballStemmer
from dataclasses import dataclass


@with_value
def echo(document_list):
    for document in document_list:
        yield document


@with_value
def to_lower(document_list):
    '''Transforma todo el texto a minusculas'''
    for document in document_list:        
        document.content = [word.lower() for word in document.content]
        yield document

word_regex = re.compile(r'\w')


@with_value
def drop_punctuations(document_list):
    '''Elimina todos los signos de puntuacion
        Caso 1.Signo de puntuacion tiene al menos un carácter de espacio
            alrededor.
            - Elimina simplemente el caracter
        Caso 2. Signo de puntuación no tiene ningun carácter de espacio
            alrededor.
            - Es asignado un espacio en lugar del signo de puntuacion.

        Nota. Debe ser corrido despues de ajustar los hashtag y/o nombres de usuarios.
        Nota. Tanto los usuarios como hashtag pueden contener el signo de puntuación
        _. Este no va ser eliminado si se encuentra en una palabra que empieza con
        $ = user|hashtag
    '''
    punctuations = ["”", "“"] + list(string.punctuation)
    for document in document_list:
        document.content = [replace_list(word, punctuations, '') for word in document.content]
        yield document


@with_value
def drop_stop_words(document_list):
    '''Remueve todas las palabras vacias del lenguaje espanol'''
    stop_words = load_stop_words()

    for document in document_list:
        document.content = [word for word in document.content if word not in stop_words]
        yield document


@with_value
def tee_and_write_to_file(document_list, fp):
    for document in document_list:
        fp.write(f'<id>{document.id}</id>\n')
        fp.write(f'<periodico>{document.newpaper}</periodico>\n')
        fp.write(f'<noticia>{document.title}</noticia>\n')
        fp.write(f'<cuerpo>\n')
        fp.write(' '.join(document.content) + "\n")
        fp.write('</cuerpo>\n')
        fp.write('\n')

        yield document


@with_value
def drop_numbers(document_list):
    remove_digits = str.maketrans('', '', digits)

    for document in document_list:
        document.content = [word.translate(remove_digits) for word in document.content]
        yield document


@with_value
def stem(document_list):
    s = SnowballStemmer(language='spanish')

    for document in document_list:
        document.content = [s.stem(word) for word in document.content]
        yield document


@dataclass
class QueryWrapper:
    content: str


def clean_query(terms):
    wrapper = QueryWrapper(terms)

    pipeline = (
        echo([wrapper])
        >> drop_numbers
        >> drop_punctuations
        >> to_lower
        >> drop_stop_words
        >> stem
    )
    return [term.content for term in pipeline.value][0]