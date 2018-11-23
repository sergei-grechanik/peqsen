
import hypothesis
from hypothesis import strategies

import peqsen.util
from peqsen import Listener, Node, Hyperedge, Hypergraph, Term, EvaluableSignature

import inspect

import lark
from lark import Lark


class _ParsedTermTransformer(lark.Transformer):
    def equality(self, x):
        ((f1, xs1), (f2, xs2)) = x
        return (lambda d: (f1(d), f2(d)), xs1 | xs2)
    def call(self, x):
        (lbl, args) = x
        #lbl = lbl.value
        return ((lambda d: Term(lbl, [a[0](d) for a in args])),
                set().union(*(a[1] for a in args)))
    def variable(self, tname):
        name = tname[0].value
        return (lambda d: d[name]), {name}
    def const(self, tlbl):
        lbl = tlbl[0].value
        return self.call((lbl, []))
    def arglist(self, xs):
        return xs

_term_parser = Lark(r"""
    ?start: term | equality
    equality: term "=" term
    ?term: call | const | variable | "(" term ")"
    call: LABEL_NAME arglist
    variable: VAR_NAME
    const: CONST_NAME
    arglist: "(" [term ("," term)*] ","? ")"

    LABEL_NAME: ("_" | LETTER | DIGIT)+
    CONST_NAME: (UCASE_LETTER | DIGIT) ("_" | LETTER | DIGIT | SYMBOL)*
    VAR_NAME: ("_" | LCASE_LETTER) ("_" | LETTER | DIGIT)*
    SYMBOL: /[$-]/

    %import common.ESCAPED_STRING
    %import common.LCASE_LETTER
    %import common.UCASE_LETTER
    %import common.LETTER
    %import common.DIGIT
    %import common.WS
    %ignore WS

    """)

def parse(text):
    f, xs = _ParsedTermTransformer().transform(_term_parser.parse(text))
    return f({x: Node() for x in xs})

def make_evaluable_sig(labels2funs, generate):
    signature = []
    functions = []
    labels2funs_pp = {}
    for l, f in labels2funs.items():
        if callable(f):
            signature.append((l, len(inspect.signature(f).parameters)))
            functions.append(f)
            labels2funs_pp[l] = f
        else:
            signature.append((l, 0))
            functions.append(lambda: f)
            labels2funs_pp[l] = lambda: f

    def evaluate(label, vals, labels2funs=labels2funs):
        return labels2funs[label](*vals)

    return EvaluableSignature(signature=signature, evaluate=evaluate,
                              generate=generate, functions=functions)

def extend_evaluable_sig(sig, labels2funs):
    new_labels2funs = {l: f for (l, _), f in zip(sig.signature, sig.functions)}
    new_labels2funs.update(labels2funs)
    return make_evaluable_sig(new_labels2funs, sig.generate)

BooleanSig = \
    make_evaluable_sig(
        {'0': False,
         '1': True,
         'neg': lambda x: not x,
         'and': lambda x, y: x and y,
         'or' : lambda x, y: x or y},
        strategies.booleans)

BooleanExtSig = \
    extend_evaluable_sig(
        BooleanSig,
        {'impl': lambda x, y: (not x) or y,
         'xor': lambda x, y: x != y,
         'eq': lambda x, y: x == y})

def comm_assoc(label, identity=None):
    return []

#  BooleanTheory = \
#      make_theory(
#          BooleanSig,
#          ["neg(0) = 1",
#           "neg(1) = 0",
#           ""])

