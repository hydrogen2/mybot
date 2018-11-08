from nltk.tag import StanfordPOSTagger
from nltk.parse import malt
from nltk.stem import WordNetLemmatizer

tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger', '/Users/weizhiwei/nlp/stanford-postagger-2018-10-16/stanford-postagger.jar')
parser = malt.MaltParser('/Users/weizhiwei/nlp/maltparser-1.9.1', '/Users/weizhiwei/nlp/engmalt.linear-1.7.mco')
stemmer = WordNetLemmatizer()

kb = {}

while True:
    line = raw_input()
    if line == 'bye':
        break
    
    tagged = tagger.tag(line.split())
    parse = parser.parse_tagged_sents([tagged]).next().next()
    
    parse.tree()
    
    // walk the tree
    // look up the words
    // ask questions
    // 