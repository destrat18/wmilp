from pysmt.shortcuts import GE, LE, LT, And, Bool, Iff, Ite, Plus, Real, Symbol, Times
from pysmt.typing import BOOL, REAL

from wmipa import WMI

# variables definition
a = Symbol("A", BOOL)
b = Symbol("B", BOOL)
c = Symbol("C", BOOL)
d = Symbol("D", BOOL)
e = Symbol("E", BOOL)
x1 = Symbol("x1", REAL)
x2 = Symbol("x2", REAL)

# formula definition
# fmt: off
phi = Bool(True)

# weight function definition
w = Ite(a & (x1 >= 1.5),
        Times(
            Ite(b | (x2 >= 1),
                x1,
                2 * x1
                ),
            Ite((x2 <= 1),
                Plus(
                    Ite(x2 <= 0.5, x1 * x1, x1 * x2),
                    Ite(x1 + x2 <= 1, x2 * x2, x1 * x2)
                ),
                Plus(
                    Ite(x2 <= 0.5, x1 * x1, x1 * x2),
                    Ite(x1 + x2 <= 1, x2 * x2, x1 * x2)
                )
                )
        ),
        Times(
            Ite((x2 <= 1.5),
                3 * (x1 * x1),
                4 * (x1)
                ),
            Ite(d & (x2 + x1 >= 1),
                3 * (x2 * x2),
                4 * (x2)
                )
        )
        )

chi = And(LE(Real(0), x1), LT(x1, Real(1)),
          LE(Real(0), x2), LT(x2, Real(2)),
          Iff(a, GE(x2, Real(1))))
# fmt: on

print("Formula:", phi.serialize())
print("Weight function:", w.serialize())
print("Support:", chi.serialize())

print()
for mode in [WMI.MODE_ALLSMT, WMI.MODE_PA, WMI.MODE_SA_PA, WMI.MODE_SAE4WMI]:
    wmi = WMI(chi, w)
    result, n_integrations = wmi.computeWMI(phi, mode=mode, cache=-1)
    print(
        "WMI with mode {} \t result = {}, \t # integrations = {}".format(
            mode, result, n_integrations
        )
    )
