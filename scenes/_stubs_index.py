from manim import *
try:
    from ._stubs import make_stub
except Exception:
    import os, sys, os.path as p
    sys.path.append(p.dirname(__file__))
    from _stubs import make_stub

# Newton’s Laws
N2L_1D               = make_stub("N2L_1D", "Newton's 2nd Law (1D)", {"F_net": 4, "m":1, "t_max":6})
InclinedPlaneFriction= make_stub("InclinedPlaneFriction", "Inclined Plane (μ_s→μ_k)", {"theta_deg":25,"mu_s":0.5,"mu_k":0.3})
AtwoodMachine        = make_stub("AtwoodMachine", "Atwood Machine", {"m1":1.0,"m2":2.0,"g":9.8,"t_max":5})

# Energy
WorkEnergy1D         = make_stub("WorkEnergy1D", "Work–Energy Theorem", {"F":3,"m":1,"s":5})
ConserveEnergyCoaster= make_stub("ConserveEnergyCoaster", "Energy Conservation (Coaster)", {"h0":6,"v0":0})
PowerOutput          = make_stub("PowerOutput", "Power vs. Time", {"F":200,"v0":3})

# Momentum
Elastic1D            = make_stub("Elastic1D", "Elastic Collision (1D)", {"m1":1,"m2":2,"v1i":3,"v2i":-1})
Inelastic1D          = make_stub("Inelastic1D", "Inelastic Collision (1D)", {"m1":1,"m2":1,"v1i":2,"v2i":-1})
Momentum2D           = make_stub("Momentum2D", "2D Collision", {"m1":1,"m2":2,"v1":3,"v2":2,"phi_deg":30})

# Circular & SHM
UniformCircular      = make_stub("UniformCircular", "Uniform Circular Motion", {"r":2,"v":4})
BankedTurn           = make_stub("BankedTurn", "Banked Curve", {"r":30,"v":20,"mu":0.0,"theta_deg":15})
MassSpringSHM        = make_stub("MassSpringSHM", "Mass–Spring SHM", {"A":0.5,"k":10,"m":1})
PendulumSmallAngle   = make_stub("PendulumSmallAngle", "Pendulum (small-angle)", {"L":1,"theta0_deg":10})
