
import lark
from lark import Lark

import peqsen.util
from peqsen import Node, Hyperedge, Term

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
    if isinstance(text, str):
        f, xs = _ParsedTermTransformer().transform(_term_parser.parse(text))
        return f({x: Node() for x in xs})
    elif isinstance(text, tuple):
        return tuple(parse(t) for t in text)
    elif isinstance(text, (Node, Hyperedge, Term)):
        return text
    else:
        raise ValueError("Value {} of type {} cannot be parsed or interpreted as anything useful"
                         .format(text, type(text)))

