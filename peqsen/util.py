
import sys

_tracing_indent = 0
_tracing_indent_str = "|   "

def printind_helper(string, indent, width, add_to_nonfirst="|       "):
    for l in string.split("\n"):
        nonfirst = 0
        while l:
            localwidth = width - len(add_to_nonfirst)*nonfirst
            if localwidth < 40:
                localwidth = 40
            print(_tracing_indent_str*indent + add_to_nonfirst*nonfirst + l[:localwidth])
            l = l[localwidth:]
            nonfirst = 1

def printind(*args, width=150):
    indent = _tracing_indent
    localwidth = width - indent * len(_tracing_indent_str)
    if len(args) == 1:
        printind_helper(str(args[0]), indent, localwidth)
    else:
        strs = [str(a) for a in args]
        joined = " ".join(strs)
        joined = " ".join(joined.split())
        printind_helper(joined, indent, localwidth)

def traced(fun):
    def wrapper(*args, **kwargs):
        global _tracing_indent
        printind(fun.__name__, args, kwargs)
        _tracing_indent += 1
        try:
            res = fun(*args, **kwargs)
        except Exception as e:
            printind("EXCEPT", e)
            _tracing_indent -= 1
            raise
        except:
            printind("EXCEPT", sys.exc_info()[0])
            _tracing_indent -= 1
            raise
        printind("return", res)
        _tracing_indent -= 1
        return res
    return wrapper

def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
