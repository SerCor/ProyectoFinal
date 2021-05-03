from collections import Counter
import numpy as np
from itertools import chain
from tf_idf import TF_IDF, tf
import math


def frequency_per_document(document):
    '''
    Genera un diccionario con las frecuencias de los terminos dentro del coumento

    Args.
    document:
        Iterable de lineas que contiene el documento

    Return.
    counter:
        Counter con las frecuencias de las palabras
    '''
    if isinstance(document, str):
        document = [document]

    counter = Counter()
    for line in document:
        counter.update(word.strip() for word in line.split())

    return counter


def generate_vector(frequency, terms):
    '''
    Genera un vector de pesos que representa a una consulta/query

    Args.
    frequency:
        Diccionario que representa las repeticiones de un termino en un documento
        o query
    terms:
        Lista de terminos que estan involucrados en el modelo.

    Return.
    vector_weigths:
        numpy array de los pesos de cada palabra. Los pesos estan ordenados
        conforme al orden de los terminos pasados en el argumento terms
    '''
    return np.array([frequency.get(k, 0) for k in terms])


def calculate_correlation(vector_d, vector_q):
    '''Calcula la correlacion entre dos vectores a traves del 
    la similitud por coseno'''
    return np.dot(vector_d, vector_q) / (np.linalg.norm(vector_d) * np.linalg.norm(vector_q))


def representation_of_documents(*documents):
    '''Retorna una representacion TF IDF de los documentos'''
    t = TF_IDF()
    t.train(documents)
    return t



def run(query_terms, *documents):
    '''Retorna un ranking de documentos a partir de una consulta'''
    # Calculate frequencys
    tf_idf = representation_of_documents(*documents)
    c = Counter(query_terms)
    query_frequency = {word: tf(c[word]) * tf_idf.inverse_frequencys.get(word, 0)
                          for word in c}

    # Calculate all the terms
    query_terms = set(query_frequency.keys())
    terms = list(query_terms.union(*[f.keys() for f in tf_idf.weigths_per_doc.values()]))

    # Calculate vectors
    vector_per_doc = {name: generate_vector(frequency, terms)
                      for (name, frequency) in tf_idf.weigths_per_doc.items()}
    query_vector = generate_vector(query_frequency, terms)
    
    # Correlations
    correlations_per_doc = {name: calculate_correlation(doc_vector, query_vector)
                    for (name, doc_vector) in vector_per_doc.items()}
    
    return correlations_per_doc