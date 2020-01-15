#!/usr/local/bin/python

import sys
import os
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/usr/local/Cellar/swi-prolog/8.0.2_1/libexec/lib/swipl/lib/x86_64-darwin/'
import pyswip
from pyswip import Prolog, registerForeign, Atom
from sympy import *
from sympy.solvers.solveset import nonlinsolve
from sympy.parsing.sympy_parser import parse_expr

pyswip.easy.PL_unify_atom_chars = pyswip.core._lib.PL_unify_atom_chars

def w_nonlinsolve(*a):
    if True: #isinstance(a[0], Atom) and isinstance(a[1], Atom):
        eqs = [parse_expr(s) for s in str(a[0]).split()]
        xs = symbols(str(a[1]))
        ans = nonlinsolve(eqs, xs)
        a[2].value = str(ans.args[0])
        return True
    else:
        return False

registerForeign(w_nonlinsolve, arity=3)

prolog = Prolog()
# prolog.consult('mybot.pl')
print(list(prolog.query("Eqs='x+y-180 a+b-20 5*a-x 15*b-y', Xs='a b x y', w_nonlinsolve(Eqs, Xs, Ans)", catcherrors=False)))