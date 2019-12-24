% for now assume all answers are number type

% solve an unknown FE of a Frame
solve(FrameName, Entity) :-
    quantifiable(FrameName), solveQuant(FrameName, Entity).

quantifiable(f:quant).
addable(f:quant).

solveQuant(FrameName, Entity) :- % known quant
    call(FrameName, F),
    f:entity(F, Entity),
    f:value(F, Val),
    makeVar(FrameName, Entity, Var),
    makeEquation(eq, Var, Val).

solveQuant(FrameName, Part) :- % addition
    addable(FrameName),
    partWhole(Part, Whole),
    partition(Whole, Parts),
    member(Part, Parts),
    maplist(makeVar(FrameName), Parts, Vars),
    makeVar(FrameName, Whole, Var),
    makeEquation(sum, Vars, Var),
    delete(Parts, Part, PartsExcludingThis),
    maplist(solveQuant(FrameName), PartsExcludingThis),
    solveQuant(FrameName, Whole).

partition(Whole, Parts) :-
    partWhole(E1, Whole),
    partWhole(E2, Whole),
    f:entity(P1, E1),
    f:entity(P2, E2),
    negatePred(P1, P2),
    Parts = [E1, E2].

makeVar(FrameName, Entity, Var) :- write('let '), write(Entity), write('.'), write(FrameName), writeln(' = x.'), Var = Entity.
makeEquation(eq, Var, Val) :- write(Var), write(' = '), writeln(Val).
makeEquation(sum, Vars, Var) :- atomic_list_concat(Vars, ' + ', L), write(L), write(' = '), writeln(Var).

partWhole(Part, Whole) :-
    f:part_whole(F),
    f:part(F, Part),
    f:whole(F, Whole).

pred(Node, Pred) :-
    f:color(Pred),
    f:entity(Pred, Node).

predsEqual(P1, P2) :-
    f:color(P1),
    f:color(P2),
    f:value(P1, C1),
    f:value(P2, C2),
    C1 = C2.

negatePred(P, NP) :- pnp(P, NP).
negatePred(NP, P) :- pnp(P, NP).

np(Pred) :- f:neg(F), f:pred(F, Pred).
pnp(Pred, NegPred) :-
    predsEqual(Pred, NegPred),
    \+ np(Pred),
    np(NegPred).

f:quant(f6). f:entity(f6, e1). f:value(f6, 8).
f:quant(f7). f:entity(f7, e2). f:value(f7, 3).
f:part_whole(f1). f:whole(f1, e1). f:part(f1, e2).
f:color(f3). f:value(f3, brown). f:entity(f3, e2).
f:part_whole(f2). f:whole(f2, e1). f:part(f2, e3).
f:color(f4). f:value(f4, brown). f:entity(f4, e3).
f:neg(f5). f:pred(f5, f4).
