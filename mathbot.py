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

def parse(words, translateBack):
    pp.pprint(words)
    parses = list(parser.parse_sents([words]).next())
    pp.pprint((len(parses), 'parses'))
    for i, parse in enumerate(parses):
        walkTree(parse, translateBack)
        pp.pprint(('Parse #'+str(i), parse.tree()))
        # walkTree(parse, )

def process(sent):
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
        # translation rules:
        # predicates like \\(a + 2b = 20\\) and \\(a \\geq 6\\) => 'it is good'
        # nouns like \\(x\\) and \\(x + 1\\) => 'apple'
        # Let the function f be defined by \\(f(x) = \\frac { x^2 } { 3 } + 6\\) also treated like nouns
        latex = matchObj.group(1) or matchObj.group(2)
        print(latex)
        expr = latex2Sympy(latex)
        if expr.is_Relational:
            placeholder = ' relExpr '
            relExprs.append(expr)
        else:
            placeholder = ' nomExpr '
            nomExprs.append(expr)
        return placeholder

    def translateBack(node):
        exprAndchildWords = wordAddr2Expr.get(node['address'])
        if exprAndchildWords is not None:
            node['word'] = exprAndchildWords[0]
            for childWord in exprAndchildWords[1:]:
                for indices in node['deps'].values():
                    try:
                        indices.remove(childWord)
                    except:
                        pass

    relExprs = []
    nomExprs = []
    sent = re.sub(r'\\\((.+?)\\\)|(\d+)', translateLatex, sent)
    words = nltk.word_tokenize(sent)
    newWords = []
    wordAddr2Expr = {}
    i = 1
    for w in words:
        if w == 'relExpr':
            newWords += ['it', 'is', 'good']
            wordAddr2Expr[i+2] = (relExprs.pop(0), i, i+1) # good should always be the root
            i += 3
        elif w == 'nomExpr':
            newWords += ['apple']
            wordAddr2Expr[i] = (nomExprs.pop(0),)
            i += 1
        else:
            newWords.append(w)
            i += 1
    parse(newWords, translateBack)

def solve(problem):
    sentences = problem.split('.')
    for sentence in sentences:
        process(sentence)

def test(f):
    with open(f) as json_file:
        data = json.load(json_file)

    problems = [x['question'] for x in data if 'closed' in x['tags']]
    problems = problems[2:3]
    pp.pprint(('test data: ', problems))
    for problem in problems:
        solve(problem)

test('sat.dev.json')