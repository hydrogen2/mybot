% for now assume all answers are number type

% solve an unknown FE of a Frame
solve(F, FE) :-
    quantifiable(F, FE), solveForQuant(F, FE).

solveForQuant(F, FE) :- % known quant
    call(FE, F, Quant),
    makeVar(F, FE, Var),
    makeEquation(eq, Var, Quant).

entity(F, E) :-
    .

solveForQuant(F, FE) :- % addition
    entity(F, Part),
    partWhole(Part, Whole),
    partition(Whole, [Parts]),
    member(Parts, Part),
    makeVar(Whole, FE, Var),
    makeVars(Parts, FE, Vars),
    makeEquation(sum, Var, Vars),
    solveForQuant(Whole, Var2),
    solveForQuant(Complement, Var1).

makeVar(_, number, var).
makeEquation(eq, Var, Number) :- write(Var), write(eq), write(Number).
makeEquation(sum, Var, Var1, Var2) :- write(Var), write(plus), write(Var1), write(eq), write(Var2).

partWhole(Part, Whole) :-
    be_subset_of(Frame),
    part(Frame, Part),
    total(Frame, Whole).

pred(Node, Pred) :-
    color(Pred),
    colorentity(Pred, Node).

predsEqual(P1, P2) :-
    color(P1),
    color(P2),
    colorcolor(P1, C1),
    colorcolor(P2, C2),
    C1 = C2.

negatePred(Pred, NegPred) :- % double neg
    notpred(_, Pred),
    predsEqual(Pred, NegPred),
    \+ notpred(_, NegPred).

negatePred(Pred, NegPred) :-
    \+ notpred(_, Pred),
    notpred(_, NegPred),
    predsEqual(Pred, NegPred).


f:number(f6).
f:number.entity(f6, node1)
f:number.number(f6, 8).
f:number(f7).
f:number.entity(f7, node2)
f:number.number(f7, 3).
f:part_whole(frame1).
f:part_whole.whole(frame1, node1).
f:part_whole.part(frame1, node2).
f:color(frame3).
f:color.color(frame3, brown).
f:color.entity(frame3, node2).
f:part_whole(frame2).
f:part_whole.whole(frame2, node1).
f:part_whole.part(frame2, node3).
f:color(frame4).
f:color.color(frame4, brown).
f:color.entity(frame4, node3).
f:neg(frame5).
f:neg.pred(frame5, frame4).