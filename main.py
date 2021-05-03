import argparse
import sys
import os
from text_processing.source import from_file
from text_processing.pipeline import (
    drop_punctuations, to_lower,
    drop_stop_words, drop_numbers,
    tee_and_write_to_file, stem, echo,
    clean_query
)
from functools import partial, wraps
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from json import dump
from parser import parse_file
from model import run


def preprocess_news(path, fd1, fd2, fd3):
    print(f'[+] Inicio de procesamiento de {path}')
    tee_and_write_lexic = partial(tee_and_write_to_file, fp=fd1)
    tee_and_write_stopwords = partial(tee_and_write_to_file, fp=fd2)
    tee_and_write_stem = partial(tee_and_write_to_file, fp=fd3)

    with open(path, mode='r', encoding='utf-8') as fp:
        news = parse_file(fp)

    base_pipeline = (
        echo(news)
        >> drop_numbers
        >> drop_punctuations
        >> to_lower
        >> tee_and_write_lexic
        >> drop_stop_words
        >> tee_and_write_stopwords
        >> stem
        >> tee_and_write_stem
    )

    return base_pipeline


def cleanup_logfiles(filename_list):
    '''Truncate the file content of the logfiles'''
    for filename in filename_list:
        open(filename, encoding='utf-8', mode='w').close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('querys', nargs='+', type=str)
    return parser.parse_args()


def main():
    args = get_args()
    news_filename = 'ProyectoF_utf8.txt'
    filename_list = (
        'resultados_parciales/1.-lexic.txt',
        'resultados_parciales/2.-drop_stopwords.txt',
        'resultados_parciales/3.-stem.txt',
        'resultados_parciales/5.-result_querys.txt'
    )
    cleanup_logfiles(filename_list)

    with open(filename_list[0], encoding='utf8', mode='a+') as fd1, \
         open(filename_list[1], encoding='utf8', mode='a+') as fd2, \
         open(filename_list[2], encoding='utf8', mode='a+') as fd3:

        # Preprocessing
        clean_news =  [new for new in preprocess_news(news_filename, fd1, fd2, fd3).value]

        with open(filename_list[3], encoding='utf8', mode='w') as file:
            for query in args.querys:
                query_terms = clean_query(query.split())
                correlation = run(query_terms , *clean_news)

                file.write(f'Consulta original: {query}\n')
                file.write(f'Representacion de la consulta: {" ".join(query_terms)}\n')
                dump(correlation, file, indent=4)
                file.write(f'\n')

    print("Los resultados parciales, asi como el ranking de documentos relevantes a las consultas, fueron guardados en la carpeta resultados_parciales")

if __name__ == '__main__':
    main()
