% for now assume all answers are number type

% solve an unknown FE in a partial Frame
solve(Frame, FE) :-
    quantifiable(Frame, FE), solveForQuant(Frame, Node, FE).

solveForQuant(Frame, Node, FE) :- % known quant
    call(Frame, F),
    call(FE, F, Quant),
    makeVar(Frame, Node, FE, Var),
    makeEquation(eq, Var, Quant).
solveForQuant(Frame, Node, FE) :- % addition
    partWhole(Node, Whole),
    pred(Node, Pred),
    negatePred(Pred, NegPred),
    partWhole(Complement, Whole),
    pred(Complement, NegPred),
    makeVar(Complement, number, Var1),
    makeVar(Whole, number, Var2),
    makeEquation(sum, Var, Var1, Var2),
    solveForQuant(Whole, Var2),
    solveForQuant(Complement, Var1).
solveForQuant(Node, Var) :-


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


number(node1, 8).
number(node2, 3).
be_subset_of(frame1).
total(frame1, node1).
part(frame1, node2).
color(frame3).
colorcolor(frame3, brown).
colorentity(frame3, node2).
be_subset_of(frame2).
total(frame2, node1).
part(frame2, node3).
color(frame4).
colorcolor(frame4, brown).
colorentity(frame4, node3).
not(frame5).
notpred(frame5, frame4).