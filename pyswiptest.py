#!/usr/local/bin/python

import sys
import os
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/usr/local/Cellar/swi-prolog/8.0.2_1/libexec/lib/swipl/lib/x86_64-darwin/'
import pyswip
from pyswip import Prolog, registerForeign, Atom
from sympy import *
from sympy.solvers.solveset import nonlinsolve
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, implicit_multiplication)

pyswip.easy.PL_unify_atom_chars = pyswip.core._lib.PL_unify_atom_chars

def w_nonlinsolve(*a):
    a0 = str(a[0])
    xforms = standard_transformations
    if '_' not in a0: # for now we use _ to tell between internal and external exprs
        xforms += implicit_multiplication
    eqs = [parse_expr(s, transformations=xforms) for s in a0.split(',')]
    x = Symbol(str(a[1])) # target
    xs = set()
    for eq in eqs:
        xs.update(eq.free_symbols)
    xs = list(xs)
    xi = xs.index(x)
    sols = nonlinsolve(eqs, xs)
    sol = sols.args[0]
    a[2].value = str(sol[xi])
    return True
    
registerForeign(w_nonlinsolve, arity=3)

prolog = Prolog()
prolog.consult('mybot.pl')

# prolog.assertz("equation('3(a+2b)-(27)')")
# prolog.assertz("question('5a+10b')")
# prolog.assertz("choice('A', '18')")
# prolog.assertz("choice('B', '35')")
# prolog.assertz("choice('C', '45')")
# prolog.assertz("choice('D', '60')")
# prolog.assertz("choice('E', '72')")

prolog.assertz("f_number(f6)")
prolog.assertz("f_entity(f6, e1)")
prolog.assertz("f_value(f6, 8)")
prolog.assertz("f_number(f7)")
prolog.assertz("f_entity(f7, e2)")
prolog.assertz("f_value(f7, 3)")
prolog.assertz("f_part_whole(f1)")
prolog.assertz("f_whole(f1, e1)")
prolog.assertz("f_part(f1, e2)")
prolog.assertz("f_color(f3)")
prolog.assertz("f_value(f3, brown)")
prolog.assertz("f_entity(f3, e2)")
prolog.assertz("f_part_whole(f2)")
prolog.assertz("f_whole(f2, e1)")
prolog.assertz("f_part(f2, e3)")
prolog.assertz("f_color(f4)")
prolog.assertz("f_value(f4, brown)")
prolog.assertz("f_entity(f4, e3)")
prolog.assertz("f_neg(f5)")
prolog.assertz("f_pred(f5, f4)")

prolog.assertz("f_number(f8)")
prolog.assertz("f_entity(f8, e4)")
prolog.assertz("f_value(f8, 20)")
prolog.assertz("f_cost(f9)")
prolog.assertz("f_entity(f9, e4)")
prolog.assertz("f_value(f9, 180)")
prolog.assertz("f_part_whole(f10)")
prolog.assertz("f_whole(f10, e4)")
prolog.assertz("f_part(f10, e5)")
prolog.assertz("f_apple(f11)")
prolog.assertz("f_entity(f11, e5)")
prolog.assertz("f_part_whole(f12)")
prolog.assertz("f_whole(f12, e4)")
prolog.assertz("f_part(f12, e6)")
prolog.assertz("f_banana(f13)")
prolog.assertz("f_entity(f13, e6)")
prolog.assertz("f_each(f15)")
prolog.assertz("f_one(f15, e7)")
prolog.assertz("f_set(f15, e5)")
prolog.assertz("f_cost(f14)")
prolog.assertz("f_entity(f14, e7)")
prolog.assertz("f_value(f14, 10)")
prolog.assertz("f_each(f17)")
prolog.assertz("f_one(f17, e8)")
prolog.assertz("f_set(f17, e6)")
prolog.assertz("f_cost(f16)")
prolog.assertz("f_entity(f16, e8)")
prolog.assertz("f_value(f16, 8)")
prolog.assertz("partition(e4, [e5, e6])")

# print(list(prolog.query("assertz(equation(a)), assertz(equation(a)), setof(X, equation(X), L)", catcherrors=False)))
# print(list(prolog.query("solveQuant(f_number, e5), findall(X, equation(X), L), atomic_list_concat(L, ',', Eqs)", catcherrors=False)))
print(list(prolog.query("solve(f_number, e6, Ans)", catcherrors=False)))
print(list(prolog.query("solve(f_number, e3, Ans)", catcherrors=False)))
# print(list(prolog.query('solve(Ans)', catcherrors=False)))