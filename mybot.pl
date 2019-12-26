% for now assume all answers are number type

% solve an unknown FE of a Frame
solve(FrameName, Entity) :-
    quantifiable(FrameName), solveQuant(FrameName, Entity).

quantifiable(f_number).
addable(f_number).

quantifiable(f_cost).
addable(f_cost).

solveQuant(FrameName, Entity) :- % known quant
    call(FrameName, F),
    f_entity(F, Entity),
    f_value(F, Val),
    makeVar(FrameName, Entity, Var),
    makeEquation(eq, Var, Val).

solveQuant(FrameName, Total) :- % multiplication
    addable(FrameName),
    makeVar(f_number, Total, Var1),
    mean(FrameName, Total, Mean),
    makeVar(FrameName, Mean, Var2),
    makeVar(FrameName, Total, Prod),
    makeEquation(product, Var1, Var2, Prod),
    solveQuant(f_number, Total),
    solveQuant(FrameName, Mean).

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

mean(FrameName, Total, Mean) :-
    f_each(F),
    f_set(F, Total),
    f_one(F, Mean).

partition(Whole, Parts) :-
    partWhole(E1, Whole),
    partWhole(E2, Whole),
    f_entity(P1, E1),
    f_entity(P2, E2),
    negatePred(P1, P2),
    Parts = [E1, E2].

makeVar(FrameName, Entity, Var) :- atomic_list_concat([Entity, FrameName], '.', Var).
makeEquation(eq, Var, Val) :- write(Var), write(' = '), writeln(Val).
makeEquation(sum, Vars, Sum) :- atomic_list_concat(Vars, ' + ', L), write(L), write(' = '), writeln(Sum).
makeEquation(product, Var1, Var2, Prod) :- write(Var1), write('*'), write(Var2), write(' = '), writeln(Prod).

partWhole(Part, Whole) :-
    f_part_whole(F),
    f_part(F, Part),
    f_whole(F, Whole).

pred(Node, Pred) :-
    f_color(Pred),
    f_entity(Pred, Node).

predsEqual(P1, P2) :-
    f_color(P1),
    f_color(P2),
    f_value(P1, C1),
    f_value(P2, C2),
    C1 = C2.

negatePred(P, NP) :- pnp(P, NP).
negatePred(NP, P) :- pnp(P, NP).

np(Pred) :- f_neg(F), f_pred(F, Pred).
pnp(Pred, NegPred) :-
    predsEqual(Pred, NegPred),
    \+ np(Pred),
    np(NegPred).

f_number(f6). f_entity(f6, e1). f_value(f6, 8).
f_number(f7). f_entity(f7, e2). f_value(f7, 3).
f_part_whole(f1). f_whole(f1, e1). f_part(f1, e2).
f_color(f3). f_value(f3, brown). f_entity(f3, e2).
f_part_whole(f2). f_whole(f2, e1). f_part(f2, e3).
f_color(f4). f_value(f4, brown). f_entity(f4, e3).
f_neg(f5). f_pred(f5, f4).

f_number(f8). f_entity(f8, e4). f_value(f8, 20).
f_cost(f9). f_entity(f9, e4). f_value(f9, 180).
f_part_whole(f10). f_whole(f10, e4). f_part(f10, e5).
f_apple(f11). f_entity(f11, e5).
f_part_whole(f12). f_whole(f12, e4). f_part(f12, e6).
f_banana(f13). f_entity(f13, e6).

f_each(f15). f_one(f15, e7). f_set(f15, e5).
f_number(f18). f_entity(f18, e7). f_value(f18, 1).
f_cost(f14). f_entity(f14, e7). f_value(f14, 10).

f_each(f17). f_one(f17, e8). f_set(f17, e6).
f_number(f19). f_entity(f19, e8). f_value(f19, 1).
f_cost(f16). f_entity(f16, e8). f_value(f16, 8).
partition(e4, [e5, e6]).