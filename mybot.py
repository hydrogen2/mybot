#!/usr/bin/python

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

test1 = [
    'Tim has 8 hamsters.'
    , '3 of them are brown.'
    , 'How many of his hamsters are not brown?'
    ] # frame: has~his
# x+y=100 a+b=6 x=4a y=5b
test2 = [
    'Winson spent $180 to buy a total of 20 pears and bananas.'
    # , 'Each pear cost 5 dollars and each banana cost 6 dollars.'
    # , 'How many pears did Winson buy?'
    ]
# Winson spent $180 to buy (a total of 20) (pears and bananas).
# buy, spend, cost, pay, price
test3 = [
    'Angela bought apples and bananas at the fruit stand.'
    , 'She bought 20 pieces of fruit and spent $11.50.'
    , 'Apples cost $.50 and bananas cost $.75 each.'
    , 'How many of each did she buy?'
    ]
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

def wsd_of(tree, node):
    head, pobj = getLink(tree, node, 'head'), getLink(tree, node, 'dep:pobj')
    if head['tag'] == 'CD': # 3 of them
        return 'Be_subset_of'
    elif pobj['tag'] == 'CD': # a total of 20
        return 'Scale_value'
    else:
        return 'Entity_association'

frames = {
    'has': {
        'Possession': {
            'Owner': 'dep:nsubj',
            'Possession': 'dep:dobj'
        }
    },
    'of': {
        'Be_subset_of': { # expressing the relationship between a part and a whole.
            'Part': 'head', #TODO: conditional: tag == CD
            'Total': 'dep:pobj'
        }
        , 'Scale_value': { # expressing the relationship between a scale or measure and a value.
            'Scale': 'head',
            'Value': 'dep:pobj'
        }
        , 'Entity_association': {
            'Entity': 'head',
            'Association': 'dep:pobj'
        }
    },
    'brown': {
        'Color': { #TODO: conditional: tag == JJ && cop
            'Color': 'self',
            'Entity': 'dep:nsubj'
        }
    },
    'spent': {
        'Commerce_pay': {
            'Buyer': 'dep:nsubj',
            'Money': 'dep:dobj'
        }
    },
    'to': {
        'Purpose': { #TODO: conditonal: infinitive present and the verb
            'Goal': 'head',
            'Means': 'head.head'
        }
    },
    'buy': {
        'Commerce_buy': {
            # 'Buyer': 'dep:nsubj',
            'Goods': 'dep:dobj'
        }
    },
    'total': {
        'Amounting_to': { # tag==NN
            'Numbers': 'head', # rel==modifier
            # 'Value': 'dep:prep.dep:pobj' #
        }
    },
    'and': { # https://universaldependencies.org/u/dep/conj.html
        'And': {
            'First': 'head',
            'Conj': 'head.dep:conj'
        }
    },
    'each': {
        'Each': {
            'Group': '' #case DET: resolve(plural(head)); case ADV: head.dep:nsubj
        }
    },
    'cost': {
        'Expensiveness': {
            'Asset': 'dep:dobj',
            'Goods': 'dep:nsubj'
        }
    }
}

def op_get(frame, params):
    node, attr, ret = frame.get(params[0]), frame.get(params[1]), params[2]
    value = node.get(attr)
    if value is None:
        return False
    frame[ret] = value
    return True

def op_makeVar(frame, params):
    frame[params[2]] = params[2]
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
        # print 'call', name
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
                    # print 'try', clause
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

model = {
    'nodes': {}
}

def getCNode(model, sentenceNo, wordNo):
    return model['nodes'][sentenceNo][wordNo]    

def findCNode(model, sent, noun=None, number=None, poss=None):
    def compareNouns(noun1, noun2):
        return stemmer.lemmatize(noun1) == stemmer.lemmatize(noun2)

    def findOwner(cnode, poss):
        try:
            frames = cnode['Possession:Possession']
        except:
            return None
        # todo: find the owner that agrees with poss
        return frames[0]['Owner']

    def compareNumbers(number1, number2):
        # todo: refine
        if number1 is not None and int(number1) > 1 and number2 == 'many':
            return True
        return number1 == number2

    for sentNo in reversed(range(sent)): # should be sent+1, hack for now!
        for _, cnode in model['nodes'][sentNo].items():
            if noun is not None and compareNouns(cnode['word'], noun):
                if poss is None or findOwner(cnode, poss) is not None:
                    # pp.pprint(cnode)
                    return cnode
            if number is not None and compareNumbers(cnode.get('number'), number):
                # pp.pprint(cnode)
                return cnode

    return None

def getLink(tree, node, link):
    def getLinkSimple(node, link):
        if link == 'self':
            return node

        if link.startswith('dep:'):
            dep = link[4:]
            target = node['deps'][dep][0] if dep in node['deps'] else None
        else:
            target = node[link]
        return tree.get_by_address(target) if target is not None else None

    for comp in link.split('.'):
        node = getLinkSimple(node, comp)
    return node

# transform and update the parse into the concept graph
def walkTree(sentenceNo, tree, model):
    def inferNumber(node):
        word = node['word']

        if word == 'them':
            return 'many'

        if node['tag'] == 'CD':
            return word

        numNode = getLink(tree, node, 'dep:num')
        if numNode is not None:
            return numNode['word']

        detNode = getLink(tree, node, 'dep:det')
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
        elif getLink(tree, node, 'dep:det') is not None:
            pass
        elif getLink(tree, node, 'dep:poss') is not None:
            possNode = getLink(tree, node, 'dep:poss')
            return findCNode(model, sentenceNo, noun=word, poss=possNode['word'])
        else:
            pass
        return cnode

    def createCNode(node):
        # decide whether to create a cnode for this node based on word, tag and rel
        # look up word and infer props
        cnode = None
        address, word, tag, rel = node['address'], node['word'], node['tag'], node['rel']
        if tag in ('NN', 'NNS', 'NNP', 'NNPS', 'PRP'):
            # check if it's an existing ref
            cnode = resolveRef(node)
            if cnode is None:
                if tag == 'PRP':
                    raise Exception('Failed to resolve word: ' + word)
                else:
                    cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                        'word': word, 'number': inferNumber(node)}
        elif rel in ('nsubj', 'dobj', 'pobj'): # for non nouns like 3 or many in the subj position
            cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                'word': word, 'number': inferNumber(node)}
        elif word in frames: # for verbs and predicates
            cnode = {#'sentenceNo': sentenceNo, 'wordNo': address,
                'word': word}
        
        if cnode is not None:
            if sentenceNo not in model['nodes']:
                model['nodes'][sentenceNo] = {}
            model['nodes'][sentenceNo][address] = cnode

    def linkCNodes(node):
        address, word, tag = node['address'], node['word'], node['tag']
        if word in frames:
            wsd = globals().get('wsd_'+word)
            if wsd is not None:
                frame = wsd(tree, node)
                roles = frames[word][frame]
            else:
                frame, roles = frames[word].iteritems().next()
            frameCNode = getCNode(model, sentenceNo, address)
            
            for roleName, roleExpr in roles.iteritems():
                # TODO: parse and evaluate roleExpr
                roleNode = getLink(tree, node, roleExpr)
                roleCNode = getCNode(model, sentenceNo, roleNode['address'])
                
                frameCNode['frame'] = frame
                frameCNode[roleName] = roleCNode

                frameRole = frame+':'+roleName
                if frameRole not in roleCNode:
                    roleCNode[frameRole] = []
                roleCNode[frameRole].append(frameCNode)

    def walkTreeRecursive(i, f):
        node = tree.get_by_address(i)
        if DEBUG:
            pp.pprint(node)
        f(node)
        deps = sorted(chain.from_iterable(node['deps'].values()))
        if deps:
            for dep in deps:
                walkTreeRecursive(dep, f)
        
    walkTreeRecursive(0, createCNode)
    walkTreeRecursive(0, linkCNodes)

def findX(sentenceNo, tree, model):
    cnode, attr = None, None
    # for now assume interrogative is at word[0]
    node = tree.get_by_address(1)
    tag, word = node['tag'], node['word']
    
    if tag not in ('WDT', 'WP', 'WP$', 'WRB'):
        raise Exception('Interrogative must be the first word in a query')

    if word == 'How':
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
                    pass # JJs other than many 
            elif headTag == 'VB':
                pass
    elif word == 'what':
        # todo
        pass
    
    return cnode, attr

def demo(test):
    # install the hack
    def parseFunc(parser, sents):
        def fixDGForTest2_0(dg):
            pprint.pprint(dg.tree())
            buy = dg.get_by_address(5)
            pears = dg.get_by_address(10)
            total = dg.get_by_address(7)
            of = dg.get_by_address(8)
            num = dg.get_by_address(9)
            buy['deps']['dobj'][0] = pears['address']
            pears['rel'] = 'dobj'
            pears['head'] = buy['address']
            total['head'] = pears['address']
            pears['deps']['nmod'] = [total['address']]
            total['rel'] = 'nmod'
            pears['deps'].pop('num', None)
            of['deps']['pobj'][0] = num['address']
            num['rel'] = 'pobj'
            num['head'] = of['address']
            return dg
        
        sentParses = parser.parse_sents(sents)
        
        if str(sents) == str(massageSent(test2[0])):
            dg = sentParses.next().next()
            return (p for p in [[fixDGForTest2_0(dg)]])
        
        return sentParses

    def massageSent(sent):
        sents = [sent]
        sents = map(lambda s: nltk.word_tokenize(s), sents)
        sents = map(lambda s: filter(lambda w: w.isalnum(), s), sents)
        return sents

    for sentNo, sent in enumerate(test):
        pp.pprint(sent)
        sents = massageSent(sent)
        parses = parseFunc(parser, sents).next()
        parses = list(parses)
        parse = parses[0]
        if DEBUG:
            pp.pprint((len(parses), 'parses'))
            pp.pprint(parse.tree())
        
        # walk the tree
        walkTree(sentNo, parse, model)
        if DEBUG:
            pp.pprint(getCNode(model, 0, 1))

        # ask clarifying questions
        # validate sent
        
        # transform to model

        if sent.endswith('?'):
            # run query
            xNode, attr = findX(sentNo, parse, model)
            if DEBUG:
                pp.pprint((xNode, attr))
            pp.pprint(solve(model, xNode, attr))
        else:
            # update kb
            pass

demo(test2)
