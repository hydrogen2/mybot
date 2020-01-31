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
from nltk.inference.discourse import DrtGlueReadingCommand, DiscourseTester
from nltk.corpus import wordnet as wn
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication)
from sympy.parsing.latex import parse_latex
from sympy import Rel

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

def walkTree(tree):
    def walkTreeRecursive(i):
        node = tree.get_by_address(i)
        if DEBUG:
            pp.pprint(node)
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            for dep in deps:
                walkTreeRecursive(dep)
        
    walkTreeRecursive(0)

def parse(sent):
    pp.pprint(sent)
    sents = [sent]
    sents = map(lambda s: nltk.word_tokenize(s), sents)
    # sents = map(lambda s: filter(lambda w: w.isalnum(), s), sents)
    parses = parser.parse_sents(sents).next()
    parses = list(parses)
    pp.pprint((len(parses), 'parses'))
    for i, parse in enumerate(parses):
        pp.pprint(('Parse #'+str(i), parse.tree()))
        walkTree(parse)

def process(text):
# substitute latex
    def latex2Sympy(latex):
        def prefix(latex):
            return latex
        def postfix(expr):
            if expr.is_Equality and expr.lhs.is_Function:
                return expr.rhs
            return expr
        latex = prefix(latex)
        expr = None
        try:
            expr = parse_latex(latex)
        except:
            pass
        if expr is None:
            print('failed to convert latex to sympy: '+latex)
        else:
            expr = postfix(expr)
        return expr

    def isExprPred(expr):
        return expr.is_Relational
    
    def isExprNominal(expr):
        return not isExprPred(expr)

    def translateLatex(matchObj):
        # predicates to good, bad and tall
        # nominals to X, Y and Z
        # note latexes after preps like by are also nominals
        latex = matchObj.group(1)
        expr = latex2Sympy(latex)
        if isExprPred(expr):
            placeholder = placeholdersForPred.next()
        elif isExprNominal(expr):
            placeholder = placeholdersForNominal.next()
        else:
            print('failed to classify expr: '+expr)
        placeholder2Expr[placeholder] = expr
        return placeholder

    placeholdersForPred = (w for w in ['good', 'bad', 'tall', 'big'])
    placeholdersForNominal = (w for w in ['X', 'Y', 'Z', 'U', 'V', 'W'])
    placeholder2Expr = {}
    text = re.sub(r'\\\((.+?)\\\)', translateLatex, text)
# parse it
    parse(text)

def answer(question):
    sentences = question.split('.')
    for sentence in sentences:
# find question (by ? and wh-word)
        # print(sentence)
        wh = re.match(r'(.*[,.]\s)?(wh.*\?)', sentence) #TODO: refine
        if wh is not None:
            ifpart = wh.group(1)
            process(ifpart)
            whpart = wh.group(2)
            process(whpart) # get question type
        else:
            process(sentence)

def test(f):
    with open(f) as json_file:
        data = json.load(json_file)

    questions = [x['question'] for x in data if 'closed' in x['tags']]
    questions = questions[:2]
    print(questions)
    # find each sentence
    for question in questions:
        answer(question)

test('sat.dev.json')