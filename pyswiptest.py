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

def solve(*a):
    eqs = [parse_expr(s) for s in str(a[0]).split(',')]
    target = parse_expr(str(a[1])+'-chi')
    eqs.append(target)
    xs = set()
    for eq in eqs:
        xs.update(eq.free_symbols)
    xs = list(xs)
    xi = xs.index(Symbol('chi'))
    sols = nonlinsolve(eqs, xs)
    sol = sols.args[0]
    a[2].value = str(sol[xi])
    return True
    
registerForeign(solve, arity=3)

prolog = Prolog()
prolog.consult('mybot.pl')

prolog.assertz("premise('3*(a + 2*b)-27')")
prolog.assertz("f_number(f0)")
prolog.assertz("f_entity(f0, e0)")
prolog.assertz("f_value(f0, '(5*a + 10*b)')")
# prolog.assertz("choice('A', '18')")
# prolog.assertz("choice('B', '35')")
# prolog.assertz("choice('C', '45')")
# prolog.assertz("choice('D', '60')")
# prolog.assertz("choice('E', '72')")
print(list(prolog.query("q_value(f0, X)")))

prolog.assertz("f_part_whole(f20)")
prolog.assertz("f_whole(f20, e20)")
prolog.assertz("f_part(f20, e21)")
prolog.assertz("f_cost(f23)")
prolog.assertz("f_entity(f23, e21)")
prolog.assertz("f_value(f23, 'x')")

prolog.assertz("f_part_whole(f21)")
prolog.assertz("f_whole(f21, e20)")
prolog.assertz("f_part(f21, e22)")
prolog.assertz("f_cost(f24)")
prolog.assertz("f_entity(f24, e22)")
prolog.assertz("f_value(f24, '(x+1)')")

prolog.assertz("f_part_whole(f22)")
prolog.assertz("f_whole(f22, e20)")
prolog.assertz("f_part(f22, e23)")
prolog.assertz("f_cost(f25)")
prolog.assertz("f_entity(f25, e23)")
prolog.assertz("f_value(f25, 10)")

prolog.assertz("f_cost(f27)")
prolog.assertz("f_entity(f27, e25)")
prolog.assertz("f_value(f27, 7)")

prolog.assertz("f_number(f28)")
prolog.assertz("f_entity(f28, e20)")
prolog.assertz("f_value(f28, 3)")

prolog.assertz("mean(e20, e25)")
prolog.assertz("partition(e20, [e21, e22, e23])")

prolog.assertz("f_cost(f26)")
prolog.assertz("f_entity(f26, e24)")
prolog.assertz("f_value(f26, '(x-1)')")
print(list(prolog.query("solveQuant(f_cost, e21)")))
print(list(prolog.query("q_value(f26, X)")))


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

prolog.assertz("f_number(f18)")
prolog.assertz("f_entity(f18, e3)")
print(list(prolog.query("q_value(f18, X)")))

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

prolog.assertz("f_cost(f19)")
prolog.assertz("f_entity(f19, e6)")
print(list(prolog.query("q_value(f19, X)")))
