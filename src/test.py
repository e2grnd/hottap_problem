from fenics import *
from mshr import *
from dolfin import Mesh
import numpy as np
import matplotlib.pyplot as plt
from utils import construct_sleeve_geometry, is_in_weld_region, UniformHeatSource

#
# Fix timeframe
T = 10.0
num_steps = 100
dt = T / num_steps
#
# Choose geometric parameters
t_wall = 0.188
t_sleeve = t_wall
t_gap = 0.02
L_sleeve = 10 * t_sleeve
L_wall = 2 * L_sleeve
#
# Choose model parameters
temp_ambient = 70.0
temp_process = 325.0
h_ambient = 9.0 * (1.0 / 3600.0) * (1.0 / 144.0)
h_ambient = Constant(h_ambient)
h_process = 48.0 * (1.0 / 3600.0) * (1.0 / 144.0) 
h_process = Constant(h_process)
rho = Constant(0.284)
c_P = Constant(0.119)
k = 31.95 * (1.0 / 3600.0) * (1.0 / 12.0) 
k = Constant(k)
# 
# Create sleeve geometry
mesh = construct_sleeve_geometry(
    t_wall=t_wall,
    t_gap=t_gap,
    t_sleeve=t_sleeve,
    L_wall=L_wall,
    L_sleeve=L_sleeve,
    refinement_parameter=20
)
#
# Define function space as first order polynomial
V = FunctionSpace(mesh, 'P', 1)
#
# Define initial conditions
u_0 = Constant(temp_process) 
u_n = interpolate(u_0, V)
#
# Define forcing function
weld_heat_source = Expression(
    '(BTU / 2) * (1 + cos((t + 5)*(pi / 5)))', degree=2, BTU=150, t=0, pi=np.pi,
)
f = UniformHeatSource(degree=2)
f.set_weld_expression(
    weld_expression=weld_heat_source,
    t_wall=t_wall,
    t_sleeve=t_sleeve,
    t_gap=t_gap,
    L_wall=L_wall,
    L_sleeve=L_sleeve
)
# Plot mesh
plt.figure()
plot(mesh)
plt.savefig('mesh.png')
#
# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
F = rho*c_P*u*v*dx + dt*k*dot(grad(u), grad(v))*dx - (rho*c_P*u_n + dt*f)*v*dx - dt*h_ambient(u - temp_ambient)*v*ds
a, L = lhs(F), rhs(F)
#
# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):
    #
    # Update current time
    t += dt
    f.update_time(t)
    #
    # Compute solution
    solve(a == L, u)
    #
    # Plot solution
    if n%10 == 0:
        plt.figure()
        c = plot(u)
        plt.colorbar(c)
        plt.savefig('test_%i.png'%n)
    #
    # Update previous solution
    u_n.assign(u)

