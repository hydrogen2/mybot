import os
from itertools import chain
import pprint
import nltk
from nltk.tag import StanfordPOSTagger
from nltk.parse.malt import MaltParser
from nltk.stem import WordNetLemmatizer
from nltk.inference.discourse import DrtGlueReadingCommand, DiscourseTester
from nltk.corpus import wordnet as wn

DEBUG = True
DEBUG = False
pp = pprint.PrettyPrinter(indent=4)

test = [
    'Tim has 8 hamsters.',
    '3 of them are brown.',
    'How many of his hamsters are not brown?']

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

# size(some set) + size(its complement set wrt the super set) = size(their super set)
# if an attr is exclusive, then set(attr) and set(not attr) are complement sets
# color is an exclusive attr
# more formally: 
# number(CNode) :- attr(CNode), set(attr), size(set)
# size(set) :- size(some super set), size(complement set wrt the super set)
# complement set :- set(not attr)

# target: condition action subtargets
# x+y=100 a+b=6 x=4a y=5b
kb = {
    'solve(node attr)': [
        ['eq(attr NUMBER)', 'makeVar(node NUMBER var)', 'solveNodeNumber(node var)']
    ]
    , 'solveNodeNumber(node var)': [
        ['get(node NUMBER number)', 'makeEquation(EQ var number)'], # known number
        ['resolveVar(node NUMBER oldvar)', 'makeEquation(EQ var oldvar)'], # known var
        ['enumNodePred(node pred)', 'solveNodeNumberWithPred(node pred var)']
    ]
    , 'solveNodeNumberWithPred(node pred var)': [
        ['get(node WHOLE whole)',
         'solveComplement(node pred whole complement)',
         'makeVar(complement NUMBER var1)',
         'makeVar(whole NUMBER var2)',
         'makeEquation(SUM var var1 var2)',
         'solveNodeNumber(whole var2)',
         'solveNodeNumber(complement var1)'
        ]
    ]
    , 'solveComplement(node pred whole complement)': [
        ['isPredExclusive(pred)',
         'negate(pred negpred)',
         'getPart(whole negpred complement)']
    ]
    , 'isPredExclusive(pred)': [
        ['eq(pred COLOR)']
    ]
}

def op_get(frame, params):
    node, attr, ret = frame.get(params[0]), frame.get(params[1]), params[2]
    value = node.get(attr)
    if value is None:
        return False
    frame[ret] = value
    return True

def op_makeVar(frame, params):
    frame[params[2]] = 'var'
    return True

def op_resolveVar(frame, params):
    return False

def op_makeEquation(frame, params):
    tmpls = {
        'eq' : '{} = {}',
        'sum': '{} + {} = {}'
    }
    eq = frame[params[0]]
    vars = [frame[p] for p in params[1:]]
    print tmpls[eq].format(*vars)
    return True

def op_eq(frame, params):
    return True

def op_enumNodePred(frame, params):
    frame[params[1]] = 'pred'
    return True

def op_negate(frame, params):
    frame[params[1]] = 'neg'
    return True

def op_getPart(frame, params):
    node, pred, ret = frame.get(params[0]), frame.get(params[1]), params[2]
    frame[ret] = node.get('parts')[0]
    return True

def solve(model, node, attr):
    def parse(str):
        ss = str.split('(')
        name, params = ss[0], ss[1]
        return name, params[:-1].split() # remove ')'

    def call(name, frame, params):
        print 'call', name
        # first try kb
        for head, body in kb.items():
            headName, callParams = parse(head)
            if headName == name:
                callFrame = {}
                outParamOffset = None
                for i, callParam in enumerate(callParams):
                    if params[i] in frame: # in param
                        callFrame[callParam] = frame[params[i]]
                    else: # out param
                        outParamOffset = i
                        break
                for clause in body:
                    print 'try', clause
                    success = True
                    callFrameCopy = callFrame.copy()
                    for action in clause:
                        actionName, actionParams = parse(action)
                        for actionParam in actionParams:
                            if actionParam.isupper(): # symbol
                                callFrameCopy[actionParam] = actionParam.lower()
                        success = call(actionName, callFrameCopy, actionParams)
                        if not success:
                            break
                    if success:
                        if outParamOffset is not None:
                            for i in range(outParamOffset, len(callParams)):
                                frame[params[i]] = callFrameCopy[callParams[i]]
                        return True
                return False
        # then try builtin
        callFrame = {}
        outParamOffset = None
        for i, param in enumerate(params):
            if param in frame: # in param
                callFrame[param] = frame[param]
            else: # out param
                outParamOffset = i
                break
        success = globals()['op_'+name](callFrame, params)
        if success:
            if outParamOffset is not None:
                for i in range(outParamOffset, len(params)):
                    frame[params[i]] = callFrame[params[i]]
        return success

    call('solve', {'node': node, 'attr': attr}, ['node', 'attr'])

# concept graph (model):
# notional words as nodes, functional words are collapsed
# nodes have (predefined) atomic attributes (e.g. types and numbers) and expected slots (e.g. subject and object)
# blur line between nodes and attributes
# edges are bidir to support association
# nodes are indexed to facilitate search
# 
model = {
    'nodes': {}
}

def getCNode(model, sentenceNo, wordNo):
    return model['nodes'][sentenceNo][wordNo]    

def findCNode(model, sent, noun=None, number=None, poss=None):
    def compareNouns(noun1, noun2):
        return stemmer.lemmatize(noun1) == stemmer.lemmatize(noun2)

    def findPoss(cnode, poss):
        # todo: refine
        has = cnode.get('has')
        if has is None or has.get('obj') is not cnode:
            return None
        return has.get('subj')

    def compareNumbers(number1, number2):
        # todo: refine
        if number1 is not None and int(number1) > 1 and number2 == 'many':
            return True
        return number1 == number2

    for sentNo in reversed(range(sent)): # should be sent+1, hack for now!
        for _, cnode in model['nodes'][sentNo].items():
            if noun is not None and compareNouns(cnode['word'], noun):
                if poss is None or findPoss(cnode, poss) is not None:
                    # pp.pprint(cnode)
                    return cnode
            if number is not None and compareNumbers(cnode.get('number'), number):
                # pp.pprint(cnode)
                return cnode

    return None

def getLink(tree, node, link, dep=None):
    if link == 'deps':
        target = node['deps'][dep][0] if dep in node['deps'] else None
    else:
        target = node[link]
    return tree.get_by_address(target) if target is not None else None

# transform and update the speech tree into the concept graph
def walkTree(sentenceNo, tree, model):
    def inferNumber(node):
        word = node['word']

        if word == 'them':
            return 'many'

        if node['tag'] == 'CD':
            return word

        numNode = getLink(tree, node, 'deps', 'num')
        if numNode is not None:
            return numNode['word']

        detNode = getLink(tree, node, 'deps', 'det')
        if detNode is not None and detNode['word'] in ('a', 'an'):
            return '1'
        
        if word.endswith('s'):
            return 'many'
        
        return None

    def resolveRef(node):
        cnode = None
        address, word, tag = node['address'], node['word'], node['tag']
        if tag == 'PRP':
            return findCNode(model, sentenceNo, number=inferNumber(node))
        elif getLink(tree, node, 'deps', 'det') is not None:
            pass
        elif getLink(tree, node, 'deps', 'poss') is not None:
            possNode = getLink(tree, node, 'deps', 'poss')
            return findCNode(model, sentenceNo, noun=word, poss=possNode['word'])
        else:
            pass
        return cnode

    def createCNode(node):
        # todo: lookup word
        # todo: correct tag
        cnode = None
        address, word, tag = node['address'], node['word'], node['tag']
        if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'PRP'):
            # check if it's an existing ref
            cnode = resolveRef(node)
            if cnode is None:
                if tag == 'PRP':
                    raise Exception('Failed to resolve word: ' + word)
                else:
                    cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                        'word': word, 'number': inferNumber(node)}
        elif tag in ('VBP', 'VBZ'):
            cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                'word': word}
        elif tag in ('CD') or word == 'many':
            ofNode = getLink(tree, node, 'deps', 'prep')
            if ofNode is not None and ofNode['word'] == 'of': # for now treat 'of sth' solely as part-whole
                cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                    'word': word, 'number': inferNumber(node)}
        elif tag in ('JJ'):
            cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                'word': word}
        else:
            pass
        
        if cnode is not None:
            if sentenceNo not in model['nodes']:
                model['nodes'][sentenceNo] = {}
            model['nodes'][sentenceNo][address] = cnode

    def linkCNodes(node):
        word, tag = node['word'], node['tag']

        if tag in ('VBP', 'VBZ'):

            cnode = getCNode(model, sentenceNo, node['address'])
            
            if node['rel'] == 'cop':
                predNode = getLink(tree, node, 'head')

                subj = getLink(tree, predNode, 'deps', 'nsubj')
                if subj is not None:
                    subjc = getCNode(model, sentenceNo, subj['address'])
                    cnode['subj'] = subjc
                    subjc[word] = cnode
                
                predc = getCNode(model, sentenceNo, predNode['address'])
                cnode['pred'] = predc
                predc[word] = cnode

                neg = getLink(tree, predNode, 'deps', 'neg')
                if neg is not None:
                    cnode['neg'] = neg['word']

            else:
                subj = getLink(tree, node, 'deps', 'nsubj')
                if subj is not None:
                    subjc = getCNode(model, sentenceNo, subj['address'])
                    cnode['subj'] = subjc
                    subjc[word] = cnode

                obj = getLink(tree, node, 'deps', 'dobj')
                if obj is not None:
                    objc = getCNode(model, sentenceNo, obj['address'])
                    cnode['obj'] = objc
                    objc[word] = cnode

        elif tag in ('IN'):
            if word == 'of' and getLink(tree, node, 'deps', 'pobj') is not None:
                sub = getLink(tree, node, 'head')
                subc = getCNode(model, sentenceNo, sub['address'])

                _super = getLink(tree, node, 'deps', 'pobj')
                superc = getCNode(model, sentenceNo, _super['address'])

                subc['whole'] = superc
                if 'parts' not in superc:
                    superc['parts'] = []
                superc['parts'].append(subc)

    def walkTreeRecursive(i):
        node = tree.get_by_address(i)
        if DEBUG:
            pp.pprint(node)
        createCNode(node)
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            for dep in deps:
                walkTreeRecursive(dep)
        linkCNodes(node)
        
    walkTreeRecursive(0)

def findX(sentenceNo, tree, model):
    cnode, attr = None, None
    # for now assume interrogative is at word[0]
    node = tree.get_by_address(1)
    tag, word = node['tag'], node['word']
    
    if tag not in ('WDT', 'WP', 'WP$', 'WRB'):
        raise Exception('Interrogative must be the first word in a query')

    if word == 'how':
        headNode = getLink(tree, node, 'head')
        if headNode is not None:
            headTag = headNode['tag']
            if headTag == 'JJ':
                headWord = headNode['word']
                if headWord == 'many':
                    xNode = getLink(tree, headNode, 'head')
                    if xNode is not None and xNode['tag'].startswith('NN'): # todo: refine to isNoun()
                        cnode, attr = getCNode(model, sentenceNo, xNode['address']), 'number'
                    else: # assume many is already made a cnode
                        cnode, attr = getCNode(model, sentenceNo, headNode['address']), 'number'
                else:
                    pass
            elif headTag == 'VB':
                pass
    elif word == 'what':
        # todo
        pass
    
    return cnode, attr

for sentNo, sent in enumerate(test):
    words = nltk.word_tokenize(sent)
    words = [word.lower() for word in words if word.isalnum()]
    parse = parser.parse_sents([words]).next().next()
    print parse.tree()
    
    # walk the tree
    walkTree(sentNo, parse, model)

    # ask clarifying questions
    # validate sent
    
    # transform to model

    if sent.endswith('?'):
        # run query
        xNode, attr = findX(sentNo, parse, model)
        pp.pprint((xNode, attr))
        pp.pprint(solve(model, xNode, attr))
    else:
        # update kb
        pass
