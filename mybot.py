import os
from nltk.tag import StanfordPOSTagger
from nltk.parse.malt import MaltParser
from nltk.stem import WordNetLemmatizer

os.environ['STANFORD_MODELS'] = 'libs/stanford-postagger-2018-10-16/models'

tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger', 'libs/stanford-postagger-2018-10-16/stanford-postagger.jar')
parser = MaltParser(os.path.dirname(os.path.abspath(__file__))+'/libs/maltparser-1.9.2', 'libs/engmalt.linear-1.7.mco')
stemmer = WordNetLemmatizer()

kb = {}

while True:
    line = raw_input()
    if line == 'bye':
        break
    
    tagged = tagger.tag(line.split())
    parse = parser.parse_tagged_sents([tagged]).next().next()
    
    print parse.tree()
    
    # walk the tree
    # look up the words
    # ask questions
    #
