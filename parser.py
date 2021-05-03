from bs4 import BeautifulSoup
from dataclasses import dataclass
from text_processing.decorators import with_value
from typing import Optional
from itertools import chain


@dataclass
class News:
    id: int
    newpaper: str
    title: str
    content: str

    def __iter__(self):
        return self.content

    def __str__(self):
        return (
            f'<periodico>{self.newpaper}</periodico>\n'
            f'<noticia>{self.title}</noticia>\n'
            f'<cuerpo>\n'
            ' '.join(self.content) + "\n"
            '</cuerpo>\n'
        )



def get_all_parts(soup):
    return (
        [p.text.strip() for p in soup.find_all('periodico')],
        [n.text.strip() for n in soup.find_all(['noticia', 'noticias'])],
        chain([c.text.strip().split() for c in soup.find_all('cuerpo')])
    )


def parse_file(fp):
    '''Retorna una lista con las noticias'''
    soup = BeautifulSoup(fp, 'html.parser')
    newpaper_list, title_list, body_list = get_all_parts(soup)
    
    return [News(n, new_paper, title, content) for n, (new_paper, title, content) in 
        enumerate(zip(newpaper_list, title_list, body_list))]