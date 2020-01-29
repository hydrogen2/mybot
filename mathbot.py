#!/usr/local/bin/python

import json
import re
import sys
import os
from itertools import chain
import pprint
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.parse.malt import MaltParser
from nltk.parse.dependencygraph import DependencyGraph
from nltk.stem import WordNetLemmatizer
from nltk.inference.discourse import DrtGlueReadingCommand, DiscourseTester
from nltk.corpus import wordnet as wn

DEBUG = True
# DEBUG = False
pp = pprint.PrettyPrinter(indent=4)

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

def latex2sympy(latex):
    return latex

def process(text):
# substitute latex
    def translateLatex(matchobj):
        return 'A'
    text = re.sub(r'\\\(.+?\\\)', translateLatex, text)
    print(text)
# parse it
    parse(text)

def answer(question):
    sentences = question.split('.')
    for sentence in sentences:
# find question (by ? and wh-word)
        # print(sentence)
        wh = re.match(r'(.*[,.]\s)?(what\s.*\?)', sentence) #TODO: refine
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
    questions = questions[:1]
    print(questions)
    # find each sentence
    for question in questions:
        answer(question)

test('sat.dev.json')