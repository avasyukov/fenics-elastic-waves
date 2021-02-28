# Based on https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html

from dolfin import *
import numpy as np

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Define mesh
max_x = 2.0
size_y = 1.0
pulse_r = 0.05

#mesh = BoxMesh(Point(0., -size_y, -size_z), Point(1., size_y, size_z), 50, 50, 50)
mesh = RectangleMesh(Point(0., -size_y), Point(max_x, size_y), 100, 100)
#mesh = UnitSquareMesh(100, 100)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return near(x[0], 0.) and on_boundary

# Sub domain for load at right end
def right(x, on_boundary):
#    return near(x[0], 1.) and abs(x[1]) < 0.1 and abs(x[2]) < 0.1 and on_boundary
    return near(x[0], max_x) and abs(x[1]) <= pulse_r and on_boundary


# Elastic parameters
E  = 1000.0
nu = 0.3
mu    = Constant(E / (2.0*(1.0 + nu)))
lmbda = Constant(E*nu / ((1.0 + nu)*(1.0 - 2.0*nu)))

# Mass density
rho = Constant(1.0)

# Rayleigh damping coefficients
eta_m = Constant(0.)
eta_k = Constant(0.)

# Generalized-alpha method parameters
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5+alpha_f-alpha_m)
beta    = Constant((gamma+0.5)**2/4.)

# Time-stepping parameters
T       = 0.075
Nsteps  = 150
dt = Constant(T/Nsteps)


# We now define the time-dependent loading. Body forces are zero and the imposed loading consists of a uniform vertical traction
# applied at the ``right`` extremity. The loading amplitude will vary linearly from :math:`0` to :math:`p_0=1` over the time interval
# :math:`[0;T_c=T/5]`, after :math:`T_c` the loading is removed. For this purpose, we used the following JIT-compiled ``Expression``.
# In particular, it uses a conditional syntax using operators ``?`` and ``:`` ::

p0 = 1.
cutoff_Tc = T/10
# Define the loading as an expression depending on t
#p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)
#p = Expression(("t <= tc ? -p0*t/tc : 0", "0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)
p = Expression(("t <= tc ? -p0 : 0", "0"), t=0, tc=cutoff_Tc, p0=p0, degree=0)

# A standard vectorial :math:`P^1` FunctionSpace is now defined for the displacement, velocity and acceleration fields. We also
# define a tensorial DG-0 FunctionSpace for saving the stress field evolution::

# Define function space for displacement, velocity and acceleration
V = VectorFunctionSpace(mesh, "CG", 1)
# Define function space for stresses
Vsig = TensorFunctionSpace(mesh, "DG", 0)

# Test and trial functions are defined and the unkown displacement (corresponding to :math:`\{u_{n+1}\}` for the current time step)
# will be represented by the Function ``u``. Displacement, velocity and acceleration fields of the previous increment
# :math:`t_n` will respectively be represented by functions ``u_old``, ``v_old`` and ``a_old``::

# Test and trial functions
du = TrialFunction(V)
u_ = TestFunction(V)
# Current (unknown) displacement
u = Function(V, name="Displacement")
# Fields from previous time step (displacement, velocity, acceleration)
u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

# We now use a ``MeshFunction`` for distinguishing the different boundaries and mark the right extremity using an ``AutoSubDomain``.
# The exterior surface measure ``ds`` is then defined using the boundary subdomains. Simple Dirichlet boundary conditions are also defined at the left extremity::

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Set up boundary condition at left end
#zero = Constant((0.0, 0.0, 0.0))
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, left)

# Python functions are now defined to obtain the elastic stress tensor :math:`\sigma` (linear isotropic elasticity), the bilinear mass and stiffness forms as well
# as the damping form obtained as a linear combination of the mass and stiffness forms (Rayleigh damping). The linear form corresponding to the work of external forces is also defined::

# Stress tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

# Mass form
def m(u, u_):
    return rho*inner(u, u_)*dx

# Elastic stiffness form
def k(u, u_):
    return inner(sigma(u), sym(grad(u_)))*dx

# Rayleigh damping form
def c(u, u_):
    return eta_m*m(u, u_) + eta_k*k(u, u_)

# Work of external forces
def Wext(u_):
    return dot(u_, p)*dss(3)

# Functions for implementing the time stepping scheme are also defined. ``update_a`` returns :math:`\{\ddot{u}_{n+1}\}`
# as a function of the variables at the previous increment and of the new displacement :math:`\{u_{n+1}\}`. The function accepts a keyword ``ufl`` so that the expressions involved can be used with UFL representations if ``True`` or with array of values if ``False`` (we will make use of both possibilities later).
# In particular, the time step ``dt`` and time-stepping scheme parameters are either ``Constant`` or floats depending on the case.
# Function ``update_v`` does the same but for the new velocity :math:`\{\dot{u}_{n+1}\}` as a function of the previous variables
# and of the new acceleration. Finally, function ``update_fields`` performs the final update at the end of the time step when the new
# displacement :math:`\{u_{n+1}\}` has effectively been computed. In this context, the new acceleration and velocities are computed
# using the vector representation of the different fields. The variables keeping track of the values at the previous increment are now assigned the new values computed for the current increment::

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        beta_ = beta
    else:
        dt_ = float(dt)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dt
        gamma_ = gamma
    else:
        dt_ = float(dt)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    u_old.vector()[:] = u.vector()

# The system variational form is now built by expressing the new acceleration :math:`\{\ddot{u}_{n+1}\}` as a function of
# the TrialFunction ``du`` using ``update_a``, which here works as a UFL expression. Using this new acceleration, the same is
# done for the new velocity using ``update_v``. Intermediate averages using parameters :math:`\alpha_m,\alpha_f` of the generalized- :math:`\alpha`
# method are obtained with a user-defined fuction ``avg``. The weak form evolution equation is then written using all these
# quantities. Since the problem is linear, we then extract the bilinear and linear parts using ``rhs`` and ``lhs``::

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# Residual
a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
res = m(avg(a_old, a_new, alpha_m), u_) + c(avg(v_old, v_new, alpha_f), u_) \
       + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
a_form = lhs(res)
L_form = rhs(res)

# Alternatively, the use of ``derivative`` can be made for non-linear problems for instance or one can also directly
# formulate the system to solve, involving the modified stiffness matrix :math:`[\bar{K}]` and the various coefficients introduced earlier.
#
# Since the system matrix to solve is the same for each time step (constant time step), it is not necessary to factorize the system at each increment.
# It can be done once and for all and only perform assembly of the varying right-hand side and backsubstitution to obtain the solution
# much more efficiently. This is done by defining a ``LUSolver`` object and asking for reusing the matrix factorization::

# Define solver for reusing factorization
K, res = assemble_system(a_form, L_form, bc)
solver = LUSolver(K, "petsc")
solver.parameters["symmetric"] = True

# We now initiate the time stepping loop. We will keep track of the beam vertical tip displacement over time as well as the different
# parts of the system total energy. We will also compute the stress field and save it, along with the displacement field, in a ``XDMFFile``.
# The option `flush_ouput` enables to open the result file before the loop is finished, the ``function_share_mesh`` option tells that only one
# mesh is used for all functions of a given time step (displacement and stress) while the ``rewrite_function_mesh`` enforces that the same mesh
# is used for all time steps. These two options enables writing the mesh information only once instead of :math:`2N_{steps}` times::

# Time-stepping
time = np.linspace(0, T, Nsteps+1)
sig = Function(Vsig, name="sigma")
res_u_file = File("test/elastodynamics-u.pvd")
res_s_file = File("test/elastodynamics-s.pvd")

# The time loop is now started, the loading is first evaluated at :math:`t=t_{n+1-\alpha_f}`. The corresponding system right-hand side is then
# assembled and the system is solved. The different fields are then updated with the newly computed quantities. Finally, some post-processing is
# performed: stresses are computed and written to the result file and the tip displacement and the different energies are recorded::

def local_project(v, V, u=None):
    """Element-wise projection using LocalSolver"""
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

for (i, dt) in enumerate(np.diff(time)):

    t = time[i+1]
    print("Time: ", t)

    # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
    p.t = t-float(alpha_f*dt)

    # Solve for new displacement
    res = assemble(L_form)
    bc.apply(res)
    solver.solve(K, u.vector(), res)

    # Update old fields with new quantities
    update_fields(u, u_old, v_old, a_old)

    # Save solution
    res_u_file << u

    # Compute stresses and save them
    local_project(sigma(u), Vsig, sig)
    res_s_file << sig

    p.t = t

