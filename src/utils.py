from mshr import *
from dolfin import Mesh, Expression, DOLFIN_EPS, CellFunction, SubDomain, refine

def construct_sleeve_geometry(
    t_wall = None,
    t_gap = None,
    t_sleeve = None,
    L_wall = None,
    L_sleeve = None,
    refinement_parameter = 16,
):
    #
    # Define the domain as a polygon
    mesh = Mesh()
    domain = Polygon(
        [
            dolfin.Point(0.0, 0.0),
            dolfin.Point(L_wall, 0.0),
            dolfin.Point(L_wall, t_wall),
            dolfin.Point((L_wall - L_sleeve), t_wall),
            dolfin.Point((L_wall - L_sleeve), (t_wall + t_gap)),
            dolfin.Point(L_wall, (t_wall + t_gap)),
            dolfin.Point(L_wall, (t_sleeve + t_wall + t_gap)),
            dolfin.Point((L_wall - L_sleeve), (t_sleeve + t_wall + t_gap)),
            dolfin.Point((L_wall - L_sleeve -  t_sleeve - t_gap), t_wall),
            dolfin.Point(0.0, t_wall),
        ],
    )
    #
    # Define weld region
    weld_subdomain = Polygon(
        [
            dolfin.Point((L_wall - L_sleeve -  t_sleeve - t_gap), t_wall),
            dolfin.Point((L_wall - L_sleeve), t_wall),
            dolfin.Point((L_wall - L_sleeve), (t_wall + t_sleeve + t_gap)),
        ]
    )
    domain.set_subdomain(1, weld_subdomain)
    #
    # Mesh
    mesh = generate_mesh(domain, refinement_parameter)
    #
    # Refine in the weld
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    c = is_in_weld_region(
        t_wall = t_wall,
        t_gap = t_gap,
        t_sleeve = t_sleeve,
        L_wall = L_wall,
        L_sleeve = L_sleeve,
    )
    class MyDomain(SubDomain):
        def inside(self, x, on_boundary):
            return c(x)
    my_domain = MyDomain()
    my_domain.mark(cell_markers, True)
    mesh = refine(mesh, cell_markers)
    #
    return mesh

class is_in_weld_region(object):
    def __init__(self, t_wall, t_sleeve, t_gap, L_wall, L_sleeve):
        #
        # Define weld width, assumed isosceles triangle
        self.weld_width = t_sleeve + t_gap
        #
        # Define the bounds of the containing square
        self.vert_ub = t_wall + self.weld_width
        self.vert_lb = self.vert_ub - self.weld_width
        self.horz_ub = L_wall - L_sleeve
        self.horz_lb = self.horz_ub - self.weld_width
    
    def __call__(self, x):
        #
        # Assess if in weld region
        return (
            self.horz_lb - DOLFIN_EPS < x[0] and
            self.horz_ub + DOLFIN_EPS > x[0] and
            self.vert_lb - DOLFIN_EPS < x[1] and
            self.vert_ub + DOLFIN_EPS > x[1]
        )

class UniformHeatSource(Expression):
    def set_weld_expression(
        self,
        weld_expression=None,
        t_wall=None,
        t_sleeve=None,
        t_gap=None,
        L_wall=None,
        L_sleeve=None
    ):
        self.weld_expression = weld_expression
        self.is_in_weld_region = is_in_weld_region(t_wall, t_sleeve, t_gap, L_wall, L_sleeve)
        return

    def eval(self, values, x):
        if self.is_in_weld_region(x):
            values[0] = self.weld_expression(x)
        else:
            values[0] = 0
        return

    def update_time(self, t):
        self.weld_expression.t = t
        return


