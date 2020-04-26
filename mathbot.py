#!/usr/local/bin/python

import json
import re
import sys
import os
from itertools import chain
from pprint import PrettyPrinter
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.parse.malt import MaltParser
from nltk.parse.dependencygraph import DependencyGraph
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication)
from sympy.parsing.latex import parse_latex

# DEBUG = True
DEBUG = False
pp = PrettyPrinter(indent=4)

# hack for subprocess.DEVNULL on python 2.7
try:
    from subprocess import DEVNULL # py3k
except ImportError:
    import os
    import subprocess
    subprocess.DEVNULL = open(os.devnull, 'wb')

os.environ['STANFORD_MODELS'] = 'libs/stanford-postagger-2018-10-16/models'

tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger', 'libs/stanford-postagger-2018-10-16/stanford-postagger.jar')
parser = MaltParser(os.path.dirname(os.path.abspath(__file__))+'/libs/maltparser-1.9.1', 'libs/engmalt.linear-1.7.mco', tagger=tagger.tag)
stemmer = WordNetLemmatizer()

def walkTree(tree, f):
    def walkTreeRecursive(i):
        node = tree.get_by_address(i)
        f(node)
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            for dep in deps:
                walkTreeRecursive(dep)
    walkTreeRecursive(0)

def parse(sent, placeholder2Expr):
    def translateBack(node):
        if node['word'] in placeholder2Expr:
            node['word'] = placeholder2Expr[node['word']]

    pp.pprint(sent)
    sents = [sent]
    sents = map(lambda s: nltk.word_tokenize(s), sents)
    # sents = map(lambda s: filter(lambda w: w.isalnum(), s), sents)
    parses = parser.parse_sents(sents).next()
    parses = list(parses)
    pp.pprint((len(parses), 'parses'))
    for i, parse in enumerate(parses):
        walkTree(parse, translateBack)
        pp.pprint(('Parse #'+str(i), parse.tree()))
        walkTree(parse, )

def process(text):
# substitute latex
    def latex2Sympy(latex):
        def fixLatex(latex):
            return latex
        latex = fixLatex(latex)
        expr = None
        try:
            expr = parse_latex(latex)
        except:
            pass
        if expr is None:
            print('failed to convert latex to sympy: '+latex)
        return expr

    def translateLatex(matchObj):
        latex = matchObj.group(1)
        expr = latex2Sympy(latex)
        placeholder = placeholders.next()
        placeholder2Expr[placeholder] = expr
        return placeholder

    placeholders = (w for w in ['good', 'bad', 'tall', 'big'] if w not in text)
    placeholder2Expr = {}
    text = re.sub(r'\\\((.+?)\\\)', translateLatex, text)
    parse(text, placeholder2Expr)

def solve(problem):
    sentences = problem.split('.')
    for sentence in sentences:
# find problem (by ? and wh-word)
        # print(sentence)
        wh = re.match(r'(.*[,.]\s)?(wh.*\?)', sentence) #TODO: refine
        if wh is not None:
            ifpart = wh.group(1)
            process(ifpart)
            whpart = wh.group(2)
            process(whpart) # get problem type
        else:
            process(sentence)

def test(f):
    with open(f) as json_file:
        data = json.load(json_file)

    problems = [x['question'] for x in data if 'closed' in x['tags']]
    problems = problems[:2]
    pp.pprint(('test data: ', problems))
    for problem in problems:
        solve(problem)

test('sat.dev.json')