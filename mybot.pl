% for now assume all answers are number type

:- dynamic partition/2.
:- dynamic equation/1.

solve(Ans) :-
    solveValue(Value),
    choice(Ans, Value).

solveValue(Value) :-
    equation(Eq),
    question(Q),
    atomic_list_concat([Q, '-', chi], EqQ),
    atomic_list_concat([Eq, EqQ], ',', Eqs),
    w_nonlinsolve(Eqs, chi, Value).

% solve an unknown FE of a Frame
solve(FrameName, Entity, Value) :-
    solveQuant(FrameName, Entity),
    findall(X, equation(X), L),
    atomic_list_concat(L, ',', Eqs),
    makeVar(FrameName, Entity, Var),
    w_nonlinsolve(Eqs, Var, Value).

valuedProp(f_number).
valuedProp(f_cost).

knownVal(FrameName, Entity, Val) :-
    valuedProp(FrameName),
    call(FrameName, F),
    f_entity(F, Entity),
    f_value(F, Val).

solveQuant(FrameName, Entity) :- % known quant
    knownVal(FrameName, Entity, Val),
    makeVar(FrameName, Entity, Var),
    makeEquation(eq, Var, Val).

solveQuant(FrameName, Entity) :- % known var
    makeVar(FrameName, Entity, Var),
    equation(Eq),
    sub_atom(Eq, _, _, _, Var).

solveQuant(FrameName, Part) :- % addition
    partWhole(Part, Whole),
    solveQuant(FrameName, Whole),
    partition(Whole, Parts),
    member(Part, Parts),
    maplist(makeVar(FrameName), Parts, Vars),
    makeVar(FrameName, Whole, Var),
    makeEquation(sum, Vars, Var),
    delete(Parts, Part, PartsExcludingThis),
    maplist(solveQuant(FrameName), PartsExcludingThis).

solveQuant(FrameName, Total) :- % multiplication
    mean(Total, Mean),
    solveQuant(FrameName, Mean),
    makeVar(f_number, Total, Var1),
    makeVar(FrameName, Mean, Var2),
    makeVar(FrameName, Total, Prod),
    makeEquation(product, Var1, Var2, Prod),
    solveQuant(f_number, Total).

solveQuant(f_number, Total) :-
    mean(Total, Mean),
    knownVal(FrameName, Mean, _),
    solveQuant(FrameName, Mean),
    makeVar(f_number, Total, Var1),
    makeVar(FrameName, Mean, Var2),
    makeVar(FrameName, Total, Prod),
    makeEquation(product, Var1, Var2, Prod),
    solveQuant(FrameName, Total).

mean(Total, Mean) :-
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

mark(X) :-
    \+ equation(X),
    assertz(equation(X)).

makeVar(FrameName, Entity, Var) :- atomic_list_concat([Entity, FrameName], '_', Var).
makeEquation(eq, Var, Val) :-
    atomic_list_concat([Var, ' - ', Val], S),
    writeln(S),
    mark(S).
makeEquation(sum, Vars, Sum) :-
    sort(Vars, Sorted),
    atomic_list_concat(Sorted, ' + ', S1),
    atomic_list_concat([S1, ' - ', Sum], S),
    writeln(S),
    mark(S).
makeEquation(product, Var1, Var2, Prod) :-
    atomic_list_concat([Var1, '*', Var2, ' - ', Prod], S),
    writeln(S),
    mark(S).

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
