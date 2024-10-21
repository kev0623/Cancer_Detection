#!/usr/bin/env python

import os
import numpy as np
from PIL import Image

import ufl
import dolfinx
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import Constant, assemble_scalar
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from ufl import ds, dx, dot, form, grad, inner, FacetNormal

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from matplotlib import pyplot as plt

#from skimage.transform import resize

# CODES = os.environ['CODES']

w_heat = 0.1
u_heat = 5.0

# Functions to identify locations of different applied boundary conditions 
# Fixed bottom 
def bottom_boundary(x):
    return np.isclose(x[1], -0.5)

# Different cases for applied fixed temperature boundary conditions
def heat_boundary_0(x):
    return np.logical_and(np.isclose(x[0], 0.5), np.logical_and(np.greater_equal(x[1], float(-1/3)-0.5*w_heat), np.less_equal(x[1], float(-1/3)+0.5*w_heat)))

def heat_boundary_1(x):
    return np.logical_and(np.isclose(x[0], 0.5), np.logical_and(np.greater_equal(x[1], -0.5*w_heat), np.less_equal(x[1], 0.5*w_heat)))

def heat_boundary_2(x):
    return np.logical_and(np.isclose(x[0], 0.5), np.logical_and(np.greater_equal(x[1], float(1/3)-0.5*w_heat), np.less_equal(x[1], float(1/3)+0.5*w_heat)))

def heat_boundary_3(x):
    return np.logical_and(np.isclose(x[1], 0.5), np.logical_and(np.greater_equal(x[0], float(1/3)-0.5*w_heat), np.less_equal(x[0], float(1/3)+0.5*w_heat)))

def heat_boundary_4(x):
    return np.logical_and(np.isclose(x[1], 0.5), np.logical_and(np.greater_equal(x[0], -0.5*w_heat), np.less_equal(x[0], 0.5*w_heat)))

def heat_boundary_5(x):
    return np.logical_and(np.isclose(x[1], 0.5), np.logical_and(np.greater_equal(x[0], float(-1/3)-0.5*w_heat), np.less_equal(x[0], float(-1/3)+0.5*w_heat)))

def heat_boundary_6(x):
    return np.logical_and(np.isclose(x[0], -0.5), np.logical_and(np.greater_equal(x[1], float(1/3)-0.5*w_heat), np.less_equal(x[1], float(1/3)+0.5*w_heat)))

def heat_boundary_7(x):
    return np.logical_and(np.isclose(x[0], -0.5), np.logical_and(np.greater_equal(x[1], -0.5*w_heat), np.less_equal(x[1], 0.5*w_heat)))

def heat_boundary_8(x):
    return np.logical_and(np.isclose(x[0], -0.5), np.logical_and(np.greater_equal(x[1], float(-1/3)-0.5*w_heat), np.less_equal(x[1], float(-1/3)+0.5*w_heat)))

def heat_boundary_bulk(x):
    w_heat = 0.4
    return np.logical_and(np.isclose(x[1], 0.5), np.logical_and(np.greater_equal(x[0], -0.5*w_heat), np.less_equal(x[0], 0.5*w_heat)))


# Function to re-organize results from FEniCSx to match image pixel ordering in matplotlib
def get_vertex_order(n_elem,V):
    xyz = V.tabulate_dof_coordinates()

    nx = n_elem+1
    ny = nx
    dx = 1.0/float(nx-1)
    dy = 1.0/float(ny-1)
    vertex_order = np.zeros(nx*ny,dtype=np.int32)
    for n in range(len(xyz[:,0])):
        x = xyz[n,0]
        y = xyz[n,1]

        j = np.abs(np.around((y + 0.5)/dy))
        i = np.abs(np.around((x + 0.5)/dx))
        vertex_order[n] = j*nx + i

    return vertex_order

# Function to locate nodes in finite element domain on boundary
def get_boundary_idx(V):
    # Gather boundary indices
    EPS=1e-6
    xyz = V.tabulate_dof_coordinates()
    boundary_idx = np.where((xyz[:,0] < -0.5 + EPS) | 
        (xyz[:,0] > 0.5 - EPS) | 
        (xyz[:,1] > 0.5 - EPS))

    return boundary_idx

# Load image and normalize based on GAN val_max
def load_image(img_fname, val_max):
    img = Image.open(img_fname)    
    img = img.rotate(270)

    img_array = np.asarray(img)
    img_array = val_max * (img_array/255.0)

    return img_array

# Function to re-order finite element result based on matplotlib image pixel order
def set_img_order(img_array, n_elem, vertex_order):
    n_vert = n_elem + 1
    img_array = img_array.reshape((n_vert*n_vert,1))[vertex_order]

    return img_array

# Function set spatially varying thermal conductivity in FEnicSx function space,
# based on a given image (img_array) 
def set_kappa(img_array, n_elem, vertex_order, V):
    img_array = set_img_order(img_array, n_elem, vertex_order)
    k = fem.Function(V, name='k')

    '''
    with k.vector.localForm() as loc:
        og_order = np.array(range(loc.local_size), dtype=np.int32)
        loc.setValues(og_order, img_array)
    '''

    og_order = np.array(range(V.dofmap.index_map.size_local,), dtype=np.int32)
    k.x.array[og_order] = img_array.squeeze()
    return k

# Function to use FEniCSx to solve the steady-state heat diffusion equation
def solve_diffusion_eqn(msh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, case):
    # Choose fixed-temperature boundary based on case
    if case == 0:
        heat_boundary = heat_boundary_0
    elif case == 1:
        heat_boundary = heat_boundary_1
    elif case == 2:
        heat_boundary = heat_boundary_2
    elif case == 3:
        heat_boundary = heat_boundary_3
    elif case == 4:
        heat_boundary = heat_boundary_4
    elif case == 5:
        heat_boundary = heat_boundary_5
    elif case == 6:
        heat_boundary = heat_boundary_6
    elif case == 7:
        heat_boundary = heat_boundary_7
    elif case == 8:
        heat_boundary = heat_boundary_8
    elif case == 'bulk':
        heat_boundary = heat_boundary_bulk

    # Fixed temperature boundary
    facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                           marker=heat_boundary)
    heat_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc_heat = fem.dirichletbc(value=ScalarType(u_heat), dofs=heat_dofs, V=V)

    A = assemble_matrix(bilinear_form, bcs=[bc_bottom, bc_heat])
    A.assemble()

    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    with b.localForm() as loc_b:
        loc_b.set(0)

    fem.petsc.apply_lifting(b, [bilinear_form], [[bc_bottom, bc_heat]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc_bottom, bc_heat])

    solver.solve(b, uh.x.petsc_vec)

    # pyvista_viewer(V, uh)

    # CANNOT USE THIS VERSION OF SOLVER DUE TO MEMORY ISSUE WITH PETSC_OPTIONS
    # Possibly for the best, as other version enables more solver elements to be
    # pre-allocated

    # Run solver
    # problem = fem.petsc.LinearProblem(a, L, bcs=[bc_bottom, bc_heat], 
    #     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    # uh = problem.solve()

    return uh

# Function to solve the steady-state heat diffusion equation for the interior (bulk)
# of a domain, with a fixed temperature at the "top" of the domain
def compute_heat_bulk(img_array, nx, ny):
    n_elem = nx - 1

    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
                        cell_type=mesh.CellType.quadrilateral)

    # Finite element type
    V = fem.functionspace(msh, ("CG", 1))

    # Dirichlet boundary conditions
    # Fixed bottom boundary
    facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                           marker=bottom_boundary)
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc_bottom = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)

    # Get vertex order for arranging image data over mesh
    vertex_order = get_vertex_order(n_elem,V)

    boundary_idx = get_boundary_idx(V)

    # Set spacially varying thermal conductivity
    k = set_kappa(img_array, n_elem, vertex_order, V)

    # Variational form
    uh = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(0))
    a = k * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    b = create_vector(linear_form)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    solver = PETSc.KSP().create(msh.comm)

    case = 'bulk' 

    uh = solve_diffusion_eqn(msh, V, k, solver, uh, bilinear_form, b, bc_bottom, img_array, n_elem, nx, vertex_order, case)

    u = uh.x.array.real.astype(np.float32)

    return u


# Function to compute the heat flux through a portion of 
# the boundary of a domain
def compute_heat_flux(uh, msh, n_facets, case):
    if case == 0:
        heat_boundary = heat_boundary_0
    elif case == 1:
        heat_boundary = heat_boundary_1
    elif case == 2:
        heat_boundary = heat_boundary_2
    elif case == 3:
        heat_boundary = heat_boundary_3
    elif case == 4:
        heat_boundary = heat_boundary_4
    elif case == 5:
        heat_boundary = heat_boundary_5
    elif case == 6:
        heat_boundary = heat_boundary_6
    elif case == 7:
        heat_boundary = heat_boundary_7
    elif case == 8:
        heat_boundary = heat_boundary_8

    # Locate facets for applied heat on boundary
    heat_facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim-1),
                                       marker=heat_boundary)

    facet_tags = dolfinx.mesh.meshtags(msh, msh.topology.dim-1, heat_facets, np.full_like(heat_facets, 1, dtype=np.int32))

    # Boundary integration term for heat facets
    ds_heat = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags, subdomain_id=1)

    # Compute heat
    flux = fem.form(dot(grad(uh), n_facets) * ds_heat)
    
    flux = assemble_scalar(flux)

    return flux


# Function to compute a high-resolution synthetic test case, which is the downsampled so that
# the resulting nominal resolution test case is not directly available in the search space
# (i.e., the exact test case cannot be found) 
def compute_heat_downsample(msh_high, V_high, k_high, solver, uh, bilinear_form, b, 
                            bc_bottom, msh_low, V_low, img_array_high, n_elem_high, n_elem_low,
                            vertex_order_high, vertex_order_low, boundary_idx_high, boundary_idx_low, case):

    # Solve complete domain, high res
    n_vert_high = n_elem_high + 1
    u_high = solve_diffusion_eqn(msh_high, V_high, k_high, solver, uh, bilinear_form, b, bc_bottom, img_array_high, n_elem_high, n_vert_high, vertex_order_high, case)

    Vtmp = fem.VectorFunctionSpace(msh_low, ('CG',1))
    u_low = fem.Function(V_low)
    u_low.interpolate(u_high)

    # Subsample boundary
    u_boundary = u_low.x.array.real[boundary_idx_low].astype(np.float32)

    #write_xyzu_to_file(u_low, V_low, 10+case)
    #write_boundary_to_file(u_low, u_boundary, V_low, 10+case)

    return u_boundary

# Function to write 2D FEniCSx solution to file in the format (x, y, u)
def write_xyzu_to_file(u, V, case):
    xyz = V.tabulate_dof_coordinates()

    u = u.x.array.real

    with open(f'u_case{case}.txt', 'w') as f:
        for i in range(len(xyz[:,0])):
            f.write(f'{xyz[i,0]} {xyz[i,1]} {u[i]}\n')

# Function to write the boundary of a 2D FEniCSx solution to file
def write_boundary_to_file(u, u_boundary, V, case):
    xyz = V.tabulate_dof_coordinates()

    u = u.x.array.real

    EPS=1e-6

    with open(f'xyzu_boundary_case{case}.txt', 'w') as f:
        for i in range(len(xyz[:,0])):
            if (xyz[i,0] < -0.5 + EPS) | (xyz[i,0] > 0.5 - EPS) | (xyz[i,1] > 0.5 - EPS):
                f.write(f'{xyz[i,0]} {xyz[i,1]} {u[i]}\n')

    np.savetxt(f'u_boundary_case{case}.txt', u_boundary)


def compute_heat_boundary(mesh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, boundary_idx, case):
    # Solve complete domain 
    u = solve_diffusion_eqn(mesh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, case)

    # Subsample boundary
    u_boundary = u.x.array.real[boundary_idx].astype(np.float32)

    #write_xyzu_to_file(u, V, case)
    #write_boundary_to_file(u, u_boundary, V, case)

    return u_boundary

def compute_heat_boundary_with_flux(mesh, n_facets, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, boundary_idx, case, num_px):
    # Solve complete domain
    u = solve_diffusion_eqn(mesh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, case)
    
    # Subsample boundary
    u_boundary = u.x.array.real[boundary_idx].astype(np.float32)

    if (num_px > 0):
        flux = compute_heat_flux(u, mesh, n_facets, case)                   
        u_boundary = np.append(u_boundary, flux * np.ones(num_px, dtype=np.float32))
    
    #write_xyzu_to_file(u, V, case)
    #write_boundary_to_file(u, u_boundary, V, case)
    
    return u_boundary

def compute_heat_boundary_temp_check(mesh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, boundary_idx, case):
    # Solve complete domain 
    u = solve_diffusion_eqn(mesh, V, k, solver, uh, bilinear_form, b, bc_bottom, img, n_elem, n_vert, vertex_order, case)

    # Subsample boundary
    u_boundary = u.x.array.real[boundary_idx].astype(np.float32)

    #write_xyzu_to_file(u, V, case+10)
    #write_boundary_to_file(u, u_boundary, V, case+10)

    return u_boundary

def compute_multi_heat_boundaries(img_array, nx, ny, num_cases):

    n_elem = nx - 1

    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
                        cell_type=mesh.CellType.quadrilateral)

    # Finite element type
    V = fem.FunctionSpace(msh, ("CG", 1))

    # Dirichlet boundary conditions
    # Fixed bottom boundary
    facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                           marker=bottom_boundary)
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc_bottom = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)

    # Get vertex order for arranging image data over mesh
    vertex_order = get_vertex_order(n_elem,V)

    boundary_idx = get_boundary_idx(V)

    k = set_kappa(img_array, n_elem, vertex_order, V)
    #k = 1.0

    # Variational form
    uh = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(0))
    a = k * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    b = create_vector(linear_form)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    solver = PETSc.KSP().create(msh.comm)

    u_array = np.array([], dtype=np.float32)
    for case_n in range(num_cases):
        u_temp = compute_heat_boundary(msh, V, k, solver, uh, bilinear_form, b, bc_bottom, img_array, n_elem, nx, vertex_order, boundary_idx, case_n)
        u_array = np.append(u_array, u_temp)

    return u_array

def compute_multi_heat_boundaries_with_flux(img_array, nx, ny, num_cases, num_px):
    
    n_elem = nx - 1
    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
            points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
            cell_type=mesh.CellType.quadrilateral)

    # Normal vectors for mesh, used to compute flux
    n_facets = FacetNormal(msh)
    
    # Finite element type
    V = fem.FunctionSpace(msh, ("CG", 1))
    
    # Dirichlet boundary conditions
    # Fixed bottom boundary
    facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
            marker=bottom_boundary)
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc_bottom = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)
    
    # Get vertex order for arranging image data over mesh
    vertex_order = get_vertex_order(n_elem,V)
    
    boundary_idx = get_boundary_idx(V)
    
    k = set_kappa(img_array, n_elem, vertex_order, V)
    #k = 1.0
    
    # Variational form
    uh = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(0))
    a = k * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx
    
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)
    
    b = create_vector(linear_form)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    
    solver = PETSc.KSP().create(msh.comm)
    
    u_array = np.array([], dtype=np.float32)
    for case in range(num_cases):
        u_temp = compute_heat_boundary_with_flux(msh, n_facets, V, k, solver, uh, bilinear_form, b, bc_bottom, img_array, n_elem, nx, vertex_order, boundary_idx, case, num_px)
        u_array = np.append(u_array, u_temp)
        
    return u_array

def compute_multi_heat_boundaries_temp_check(img_array, nx, ny, num_cases):

    n_elem = nx - 1

    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
                        cell_type=mesh.CellType.quadrilateral)

    # Finite element type
    V = fem.FunctionSpace(msh, ("CG", 1))

    # Dirichlet boundary conditions
    # Fixed bottom boundary
    facets = mesh.locate_entities_boundary(msh, dim=(msh.topology.dim - 1),
                                           marker=bottom_boundary)
    boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
    bc_bottom = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)

    # Get vertex order for arranging image data over mesh
    vertex_order = get_vertex_order(n_elem,V)

    boundary_idx = get_boundary_idx(V)

    k = set_kappa(img_array, n_elem, vertex_order, V)
    #k = 1.0

    # Variational form
    uh = fem.Function(V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(msh, PETSc.ScalarType(0))
    a = k * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    b = create_vector(linear_form)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    solver = PETSc.KSP().create(msh.comm)

    u_array = np.array([], dtype=np.float32)
    for case_n in range(num_cases):
        u_temp = compute_heat_boundary_temp_check(msh, V, k, solver, uh, bilinear_form, b, bc_bottom, img_array, n_elem, nx, vertex_order, boundary_idx, case_n)
        u_array = np.append(u_array, u_temp)

    return u_array

def compute_multi_heat_boundaries_downsample(img_array_high, nx_high, ny_high, nx_low, ny_low, num_cases):

    #img_array_high = resize(img_array, (nx_high, ny_high))
    n_elem_high = nx_high - 1
    n_elem_low = nx_low - 1

    msh_high = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem_high, n_elem_high),
                        cell_type=mesh.CellType.quadrilateral)

    # Finite element type
    V_high = fem.FunctionSpace(msh_high, ("CG", 1))

    # Dirichlet boundary conditions
    # Fixed bottom boundary
    facets = mesh.locate_entities_boundary(msh_high, dim=(msh_high.topology.dim - 1),
                                           marker=bottom_boundary)
    boundary_dofs = fem.locate_dofs_topological(V=V_high, entity_dim=1, entities=facets)
    bc_bottom = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V_high)

    # Get vertex order for arranging image data over mesh
    vertex_order_high = get_vertex_order(n_elem_high, V_high)

    boundary_idx_high = get_boundary_idx(V_high)

    k_high = set_kappa(img_array_high, n_elem_high, vertex_order_high, V_high)
    #k_high = 1.0

    # Variational form
    uh = fem.Function(V_high)
    u = ufl.TrialFunction(V_high)
    v = ufl.TestFunction(V_high)
    f = fem.Constant(msh_high, PETSc.ScalarType(0))
    a = k_high * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    b = create_vector(linear_form)

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    solver = PETSc.KSP().create(msh_high.comm)

    #msh_low = mesh.create_rectangle(comm=MPI.COMM_WORLD,
    #                    points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem_low, n_elem_low),
    #                    cell_type=mesh.CellType.quadrilateral)

    msh_low = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem_low, n_elem_low),
                        cell_type=mesh.CellType.triangle)

    # Finite element type
    V_low = fem.FunctionSpace(msh_low, ("CG", 1))

    # Get vertex order for arranging image data over mesh
    vertex_order_low = get_vertex_order(n_elem_low, V_low)

    boundary_idx_low = get_boundary_idx(V_low)

    u_array = np.array([], dtype=np.float32)
    for case_n in range(num_cases):
        u_temp = compute_heat_downsample(msh_high, V_high, k_high, solver, uh, bilinear_form, b, bc_bottom, msh_low, V_low, img_array_high, n_elem_high, n_elem_low, vertex_order_high, vertex_order_low, boundary_idx_high, boundary_idx_low, case_n)
        u_array = np.append(u_array, u_temp)

    return u_array

def compute_multi_heat_boundaries_downsample_temp_check(img_array, nx_high, ny_high, nx_low, ny_low, num_cases):
    
    u_array = compute_multi_heat_boundaries_temp_check(img_array, nx_low, ny_low, num_cases)

    return u_array

def create_all_cases():
    val_min = 1
    val_max = 10
    img_fname = '{}/gan_inference/data_gen/Python/olson_data/{}_{}_28x28/matrix_{}.png'.format(CODES, val_max, val_min, 1)
    img = load_image(img_fname, val_max)

    n_vert = 28
    n_elem = n_vert-1

    #msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
    #                        points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
    #                        cell_type=mesh.CellType.quadrilateral)

    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
                            cell_type=mesh.CellType.triangle)

    # Finite element type
    V = fem.FunctionSpace(msh, ("CG", 1))

    # Get vertex order for arranging image data over mesh
    vertex_order = get_vertex_order(n_elem,V)

    for case in range(9):
        u, msh = solve_diffusion_eqn(msh, V, img, n_elem, n_vert, vertex_order, case)
        data_fname = f'u_case_{case}.xdmf'
        with io.XDMFFile(msh.comm, data_fname, "w") as file:
            file.write_mesh(msh)
            file.write_function(u)

def test_downsampling():
    val_min = 1
    val_max = 10
    img_fname = '{}/gan_inference/data_gen/Python/olson_data/{}_{}_28x28/matrix_{}.png'.format(CODES, val_max, val_min, 1)
    img = load_image(img_fname, val_max)

    n_vert = 28
    n_elem = n_vert-1

    nx_high = n_vert * 2
    ny_high = nx_high
    nx_low = n_vert
    ny_low = nx_low
    u_array = compute_multi_heat_boundaries_downsample(img, nx_high, ny_high, nx_low, ny_low, 9)

    # np.savetxt('all_cases_downsample_pre.txt',u_array)

def test_multi_heat_compute():
    val_min = 1
    val_max = 10
    img_fname = '{}/gan_inference/data_gen/Python/olson_data/{}_{}_28x28/matrix_{}.png'.format(CODES, val_max, val_min, 1)
    img = load_image(img_fname, val_max)

    n_vert = 28
    nx = n_vert
    n_elem = n_vert-1

    nx_high = n_vert * 2
    ny_high = nx_high
    nx_low = n_vert
    ny_low = nx_low
    u_array = compute_multi_heat_boundaries(img, nx, nx, 9)

    # np.savetxt('all_cases_normal_pre.txt',u_array)    


def main():
    val_min = 1
    val_max = 10
    img_fname = '{}/gan_inference/data_gen/Python/olson_data/{}_{}_28x28/matrix_{}.png'.format(CODES, val_max, val_min, 1)
    img = load_image(img_fname, val_max)

    n_vert = 28
    n_elem = n_vert-1

    msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((-0.5, -0.5), (0.5, 0.5)), n=(n_elem, n_elem),
                            cell_type=mesh.CellType.triangle)

    # Finite element type
    V = fem.FunctionSpace(msh, ("CG", 1))

    vertex_order = get_vertex_order(n_elem, V)

    # Try a random case
    case = 4
    u, msh = solve_diffusion_eqn(msh, V, img, n_elem, n_vert, vertex_order, case)

def pyvista_viewer(V, u):
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = u.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.show_axes()
    plotter.camera_position = 'xy'
    # warped = grid.warp_by_scalar()
    # plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        # plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()


if __name__ == '__main__':
    # main()
    for i in range(500):
        test_downsampling()
        test_multi_heat_compute()



