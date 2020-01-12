#!/usr/bin/python

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

parse(sys.argv[1])
