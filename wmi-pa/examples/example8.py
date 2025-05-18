from pysmt.shortcuts import GE, LE, And, Bool, Ite, Real, Symbol
from pysmt.typing import BOOL, REAL

from wmipa import WMI

# variables definition
A = Symbol("A", BOOL)
B = Symbol("B", BOOL)
C = Symbol("C", BOOL)
D = Symbol("D", BOOL)
E = Symbol("E", BOOL)
F = Symbol("F", BOOL)
G = Symbol("G", BOOL)
H = Symbol("H", BOOL)
I = Symbol("I", BOOL)
x = Symbol("x", REAL)

# formula definition
# fmt: off
phi = Bool(True)

w = Ite(And(A, B),
        Ite(And(C, D),
            Ite(And(E, F),
                Real(1.0),
                Real(1.0)
                ),
            Ite(And(F, G),
                Real(1.0),
                Real(1.0)
                )
            ),
        Ite(And(D, E),
            Ite(And(G, H),
                Real(1.0),
                Real(1.0)
                ),
            Ite(And(H, I),
                Real(1.0),
                Real(1.0)
                )
            )
        )

chi = And(GE(x, Real(-2)), LE(x, Real(2)))
# fmt: on

print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

print()
for mode in [WMI.MODE_PA, WMI.MODE_SA_PA, WMI.MODE_SAE4WMI]:
    wmi = WMI(chi, w)
    result, n_integrations = wmi.computeWMI(phi, mode=mode)
    print(
        "WMI with mode {} \t result = {}, \t # integrations = {}".format(
            mode, result, n_integrations
        )
    )
