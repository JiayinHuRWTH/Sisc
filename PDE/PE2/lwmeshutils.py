# -*- coding: utf-8 -*-
"""
Lightweight helper methods for using GMSH mesh files without the GMSH package. This module contains several useful methods for working with finite elements. 

Intended to be used for the CES/SiSc course "Numerical Methods for PDEs"

@author Alexander Fleming; alexander.fleming@rwth-aachen.de
"""
import numpy as np


def unit_interval_quadrature():
    """
    Returns quadrature points and weights for a gaussian quadrature on [-1,1].
    """
    qp = np.zeros((3, 1))
    qp[0] = -np.sqrt(3/5)
    qp[2] = np.sqrt(3/5)

    qw = np.zeros((3, 1))
    qw[0] = 5/9
    qw[1] = 8/9
    qw[2] = 5/9

    return qp, qw, 3


def unit_triangle_quadrature():
    """
    Quadrature points and weights for the Gaussian quadrature on the reference element. Exact on polynomials up to degree 5.

    From D. Braess - "Finite Elements, Theory, Fast Solvers, and Applications in Solid Mechanics", p. 99

    Returns
    -------
    quadPoints : 7x2 ndarray of floats
        quadrature points for the reference triangle stored as row vectors
    quadWeights : 7x1 array of floats
        weights for the given quadrature points
    int
        number of quadrature points used
    """
    quadPoints = np.zeros((7, 2))
    # points in the xi direction
    quadPoints[0][0] = 1/3
    quadPoints[1][0] = (6 + np.sqrt(15))/21
    quadPoints[2][0] = (9 - 2*np.sqrt(15))/21
    quadPoints[3][0] = (6 + np.sqrt(15))/21
    quadPoints[4][0] = (6 - np.sqrt(15))/21
    quadPoints[5][0] = (9 + 2*np.sqrt(15))/21
    quadPoints[6][0] = (6 - np.sqrt(15))/21
    # eta-values
    quadPoints[0][1] = 1/3
    quadPoints[1][1] = (6 + np.sqrt(15))/21
    quadPoints[2][1] = (6 + np.sqrt(15))/21
    quadPoints[3][1] = (9 - 2*np.sqrt(15))/21
    quadPoints[4][1] = (6 - np.sqrt(15))/21
    quadPoints[5][1] = (6 - np.sqrt(15))/21
    quadPoints[6][1] = (9 + 2*np.sqrt(15))/21
    # weights
    quadWeights = np.zeros((7, 1))
    quadWeights[0] = 9/80
    quadWeights[1:4] = (155 + np.sqrt(15))/2400
    quadWeights[4:7] = (155 - np.sqrt(15))/2400

    return quadPoints, quadWeights, 7


class MeshUtilities:
    """Contains many useful methods for dealing with 2-d triangular meshes"""

    def __init__(self, meshfile):
        """
        Loads the mesh from disk. 
        Parameters
        ----------
        meshfile : file path
            location of an ASCII gmesh2 file.

        Returns
        -------
        A MeshOperations object that has utility methods for use on the provided mesh file.

        """
        self.mesh = MeshGrid(meshfile)

    def reset(self):
        """
        Reloads the mesh file from disk. 
        """
        self.mesh.reload()

    def __repr__(self):
        bbinf = "Mesh bounded by ({:4f},{:4f},{:4f}) and ({:4f},{:4f},{:4f}), with:".format(*self.mesh.bounding_box[0],
                                                                                            *self.mesh.bounding_box[1])
        nodeinf = "\t{} total nodes".format(self.mesh.num_nodes)
        lineinf = "\t{} total lines and {} second order lines".format(
            self.mesh.num_lines, self.mesh.num_lines3)
        triinf = "\t{} total triangles and {} second order triangles".format(
            self.mesh.num_tris, self.mesh.num_tris6)
        quadinf = "\t{} total quads".format(self.mesh.num_quads)
        return "\n".join([bbinf, nodeinf, lineinf, triinf, quadinf])

    def get_line_tag(self, i):
        """
        Parameters
        ----------
        `i` : `int`
            the edge element whose tag we want. `i` is zero-indexed

        Returns
        -------
        `int`
            1 for inner edges \n
            2 for outer edge with a dirichlet boundary value \n
            3 for an outer edge with a von Neumann boundary value

        """
        return self.mesh.lines[i][2]

    def get_triangle_tag(self, i):
        """
        Returns the tag of triangle i. i is zero-indexed.

        Parameters
        ----------
        i : integer
            the triangle whose tag we desire.

        Returns
        -------
        integer
            1 for inner surface,
            2 for outer surface with a dirichlet boundary value,
            3 for an outer surface with a von Neumann boundary value

        """
        return self.mesh.tris[i][3]

    def get_node_count(self):
        """
        Returns
        -------
        int
            number of nodes in this mesh.

        """
        return self.mesh.num_nodes

    def get_node_list(self):
        """
        Returns
        -------
        numpy ndarray
            an nx2 array of row vectors for the position of the n nodes
            in this mesh.

        """
        return self.mesh.node_positions[:, 0:2]

    def get_triangle_count(self):
        """
        Returns
        -------
        integer
            number of triangles in this mesh.

        """
        return self.mesh.num_tris

    def get_triangle_list(self):
        """
        Returns
        -------
        numpy ndarray
            an nx3 array of the n triangles in this mesh
            each row vector is the node numbers of the nth triangle

        """
        return self.mesh.tris[:, 0:3]

    def get_triangle_nodes(self, triIdx, order):
        """
        Parameters
        ----------
        triIdx : int
            the triangle whose nodes we wish to know.
        order : int
            the order of the triangle element. n-order nodes support all lower orders
            currently can only read orders 1 and 2 from file

        Returns
        -------
        numpy.ndarray
            list of node numbers for the triangle.
        """
        if order == 1:
            return self.mesh.tris[triIdx, 0:3]
        return self.mesh.tris6[triIdx, 0:6]

    def get_quad_count(self):
        """
        Returns
        -------
        integer
            number of quadrilaterals in this mesh
        """
        return self.mesh.num_quads

    def get_quad_list(self):
        """
        Returns
        -------
        numpy.ndarray
            an nx4 array of the n quadrilateral elements in this mehs
            each row vector is the node numbers in the nth quadrilateral
        """
        return self.mesh.quads[:, 0:4]

    def get_line_count(self):
        """
        Returns
        -------
        integer
            number of lines in this mesh.

        """
        return self.mesh.num_lines

    def get_line_nodes(self, lineIdx, order):
        """
        Parameters
        ----------
        lineIdx : int
            the line whose nodes we wish to know.
        order : int
            the order of the line element. n-order nodes support all lower orders
            currently can only read orders 1 and 2 from file

        Returns
        -------
        numpy.ndarray
            list of node numbers for the line.

        """
        if order == 1:
            return self.mesh.lines[lineIdx, 0:2]
        return self.mesh.lines3[lineIdx, 0:3]

    def line_jacobian(self, lineIdx):
        """

        Parameters
        ----------
        lineIdx : integer
            the line element whose jacobian we wish to calculate.

        Returns
        -------
        J : 1x2 numpy.ndarray
            vector storing J11 and J22.

        """
        endpoints = self.mesh.lines[lineIdx][0:2]
        points = self.mesh.node_positions[endpoints][0:2]

        J = np.zeros(2)
        J[0] = 0.5 * (points[1][0] - points[0][0])
        J[1] = 0.5 * (points[1][1] - points[0][1])
        return J.reshape((1, 2))

    def triangle_jacobian(self, triIdx):
        """
        Calculates the jacobian of the transformation from the reference triangle
        to triangle triIdx
        Parameters
        ----------
        triIdx : int
            the triangle element whose jacobian we wish to calculate.

        Returns
        -------
        J : 2x2 numpy.ndarray
            The Jacobian matrix dx_i/dξ_j

        """
        vertexIds = self.mesh.tris[triIdx][0:3]
        points = self.mesh.node_positions[vertexIds][0:3]

        J = np.zeros((2, 2))
        J[0][0] = points[1][0] - points[0][0]
        J[1][0] = points[1][1] - points[0][1]
        J[0][1] = points[2][0] - points[0][0]
        J[1][1] = points[2][1] - points[0][1]

        return J

    def quad_jacobian(self, quadIdx):
        """
        Calculates the pair of jacobians to transform from the reference quadrilateral to quadrilateral quadIdx

        Since the transformation from the reference quadrilateral [(0,0), (1,0), (1,1), (0,1)] requires the transformation of two triangles, this method returns two jacobian matrices.

        Parameters
        ----------
        quadIdx : int
            the quadrilateral element whose jacobian we wish to calculate

        Returns
        -------
        J1, J2 : 2x2 numpy.ndarray
            Jacobian matrices
        """

        vertIds = self.mesh.quads[quadIdx, 0:4]
        v = self.mesh.node_positions[vertIds, 0:3]

        J1 = np.zeros((2, 2))
        J2 = np.zeros((2, 2))

        J1[0][0] = v[1][0] - v[0][0]
        J1[1][0] = v[1][1] - v[0][1]
        J1[0][1] = v[3][0] - v[0][0]
        J1[1][1] = v[3][1] - v[0][1]

        J2[0][0] = v[2][0] - v[3][0]
        J2[0][1] = v[2][0] - v[1][0]
        J2[1][0] = v[2][1] - v[3][1]
        J2[1][1] = v[2][1] - v[1][1]

        return J1, J2

    def triangle_inverse_jacobian_T(self, triIdx):
        """
        Find the inverse of the Jacobian matrix for triangle `triIdx`

        Parameters
        ----------
        `triIdx` : `int`

        Returns
        -------
        `numpy.ndarray`
            J^-1

        """
        return np.linalg.inv(self.triangle_jacobian(triIdx).T)

    def line_jacobian_determinant(self, lineIdx):
        J = self.line_jacobian(lineIdx)
        return np.sqrt(np.sum(np.power(J, 2)))

    def triangle_jacobian_determinant(self, triIdx):
        return np.linalg.det(self.triangle_jacobian(triIdx))

    def calc_triangle_integration_point(self, triIdx, int_point):
        """
        Parameters
        ----------
        triIdx : int
        int_point : ndarray
            2 element array for matrix dot product with jacobian

        Returns
        -------
        ndarray with two elements describing the transformed quadrature point
        for the purposes of evaluating the forcing function

        """
        J = self.triangle_jacobian(triIdx)
        p1n = self.mesh.tris[triIdx][0]
        refPos = self.mesh.node_positions[p1n][0:2].reshape((2, 1))

        tv = np.dot(J, int_point.reshape(2, 1))
        return np.add(refPos, tv)

    def find_L2_error_via_fourier(self, u, order, phi=None):
        """
        finds the L2 norm error between u and the 30-element Fourier series solution
          of -laplace u = 1
        this function is only useful for programming exercise 2, where the domain
          has Dirichlet B.C. on x = 0 and y = 0,1 but v.N. B.C. on x = 1

        Use for the error analysis portion of the assignment.


        Parameters
        ----------
        u : numpy ndarray
            solution vector
        order : integer
            order of the solution.
        phi : basis functions
            array of functions phi_n(x, y) with length equal to 3*order.

        Returns
        -------
        l2err : 1-element array
            l2 error of solution u.

        """
        if not phi:
            phi = [
                lambda x, y: 1-x-y,
                lambda x, y: x,
                lambda x, y: y
            ]
        if 3*order != len(phi):
            return None
        l2err = 0
        quadp, quadw, nquad = unit_triangle_quadrature()
        for i in range(self.get_triangle_count()):
            detJ = self.triangle_jacobian_determinant(i)

            for j in range(nquad):
                shape = np.zeros((len(phi), 1))
                for bfunc in range(len(phi)):
                    shape[bfunc] = phi[bfunc](quadp[j, 0], quadp[j, 1])
                mIP = self.calc_triangle_integration_point(i, quadp[j, :])
                u_exact = 0
                fo_ord = 30
                # using a fourier series to find the exact solutions for comparison
                for k in range(1, fo_ord+1):
                    for l in range(1, fo_ord+1):
                        if (l % 2) != 0:
                            # basis function for neumann BC at 1 face
                            coeff = 16/((np.pi)**4) * 1 / \
                                (l**3 * (2*k-1) + l*(2*k-1)**3/4)
                            u_exact = u_exact + coeff * \
                                np.sin((k-0.5)*np.pi *
                                       mIP[0]) * np.sin(l*np.pi*mIP[1])
                connectivity = self.get_triangle_nodes(i, order)
                # transpose shape to get a 1x1 matrix out of a (3*order x 1) and a (3*order x 1 )
                #  matrix multiplied together
                local_error = np.power(
                    np.dot(shape.flatten(), u[connectivity].flatten())-u_exact, 2)
                l2err = l2err + quadw[j]*detJ*local_error
        l2err = np.sqrt(l2err)
        return l2err

    def plot(self, ax, u, cmap='viridis'):
        """

        Parameters
        ----------
        ax : matplotlib axes that support trisurf-ing
        u : solution vector to plot; with number of elements equal 
            to the number of nodes in this mesh

        cmap : the color map to use to make the 3d plot pretty; 
               defaults to 'viridis'

        Returns
        -------
        the trisurf plot

        """

        return ax.plot_trisurf(self.get_node_list()[:, 0].flatten(),
                               self.get_node_list()[:, 1].flatten(),
                               u.flatten(), cmap=cmap)


# reference constant for the loading procedure
NODES_PER_ELEMENT_TYPE = [2, 3, 4, 4, 8, 6, 5,
                          3, 6, 9, 10, 27, 18, 14, 1, 8, 20, 15, 13]


class MeshGrid:
    """
    Describes a GMesh mesh. This is a low-level structure to keep track of small details about the mesh.
    Maintins element tag and connectivity information.
    """

    def __init__(self, file):
        self.filename = file
        self.bounding_box = np.zeros((2, 3))
        self.initialized = self.reload()

    def reload(self):
        """
        Load the mesh file from disk. 

        (perhaps again if it has been modified in the program)

        Returns
        -------
        bool
            True if the v2 ASCII mesh file was successfully loaded.

        """

        # helper function for scanf-like behavior
        def scan_for_keyword(file, keyword):
            tline = file.readline().lower().strip()
            while tline != keyword.lower().strip():
                if not tline:
                    return False
                tline = file.readline().lower().strip()
            return True

        with open(self.filename, 'r') as meshfile:
            # scan file until we reach a mesh format declarator
            if not scan_for_keyword(meshfile, "$meshformat"):
                return False
            # read mesh format information
            self.meshformat = meshfile.readline()
            # check for end of mesh formatting block
            if meshfile.readline().lower().strip() != "$endmeshformat":
                print("Can only read ASCII meshes.")
                return False

            if not scan_for_keyword(meshfile, "$nodes"):
                return False

            self.num_nodes = int(meshfile.readline())
            self.node_positions = np.zeros((self.num_nodes, 3))
            nodeids = [0]*self.num_nodes
            for i in range(self.num_nodes):
                nodeinf = meshfile.readline().split()
                # shift to zero-indexing from gmsh/matlab 1-indexing
                nodeids[i] = int(nodeinf[0]) - 1
                nodex = np.array([float(k) for k in nodeinf[1:]])
                # set axis-aligned bounding box for the mesh
                if (i == 0):
                    self.bounding_box[0] = nodex
                    self.bounding_box[1] = nodex
                else:
                    self.bounding_box[0] = [
                        min(self.bounding_box[0][k], nodex[k]) for k in range(3)]
                    self.bounding_box[1] = [
                        max(self.bounding_box[1][k], nodex[k]) for k in range(3)]
                self.node_positions[i] = nodex
            if not scan_for_keyword(meshfile, "$endnodes"):
                return False
            if not scan_for_keyword(meshfile, "$elements"):
                return False

            self.num_elements = int(meshfile.readline())
            # constants given by the file format
            num_infos = 4
            tagidx = 3
            self.element_infos = [[0]*num_infos]*self.num_elements
            self.element_tags = [0]*self.num_elements
            self.num_points = 0
            self.num_lines = 0
            self.num_tris = 0
            self.num_quads = 0
            # self.num_tets = 0
            # self.num_hexas = 0
            # self.num_prisms = 0
            # self.num_pyramids = 0
            self.num_lines3 = 0
            self.num_tris6 = 0

            self.points = np.zeros((self.num_elements, 2), np.int32)
            self.lines = np.zeros((self.num_elements, 3), np.int32)
            self.tris = np.zeros((self.num_elements, 4), np.int32)
            self.quads = np.zeros((self.num_elements, 5), np.int32)
            # self.tets = np.zeros((self.num_elements,5), np.int32)
            # self.hexas = np.zeros((self.num_elements,9), np.int32)
            # self.prisms = np.zeros((self.num_elements,7), np.int32)
            # self.pyramids = np.zeros((self.num_elements,6), np.int32)
            self.lines3 = np.zeros((self.num_elements, 4), np.int32)
            self.tris6 = np.zeros((self.num_elements, 7), np.int32)

            tokens = []
            tline = meshfile.readline().lower().strip()
            while tline != "$endelements":
                if not tline:
                    return False
                tokens = tokens + [int(k) for k in tline.split()]
                tline = meshfile.readline().lower().strip()
            for i in range(self.num_elements):
                self.element_infos[i] = [
                    tokens.pop(0) for k in range(num_infos)]
                # I have honestly no clue what this means, but it consumes tokens
                #   so it's staying in the code
                self.element_tags[i] = [tokens.pop(
                    0) for k in range(self.element_infos[i][2]-1)]
                # minus 1s to shift from one-indexing to zero-indexing
                element_nodes = [tokens.pop(
                    0)-1 for k in range(NODES_PER_ELEMENT_TYPE[self.element_infos[i][1]-1])]

                if self.element_infos[i][1] == 15:
                    self.points[self.num_points][0] = nodeids[element_nodes[0]]
                    self.points[self.num_points][1] = self.element_infos[i][tagidx]
                    self.num_points = self.num_points + 1
                elif self.element_infos[i][1] == 1:
                    self.add_line(i, nodeids, element_nodes, 1)
                elif self.element_infos[i][1] == 8:
                    self.add_line(i, nodeids, element_nodes, 2)
                elif self.element_infos[i][1] == 2:
                    self.add_triangle(i, nodeids, element_nodes, 1)
                elif self.element_infos[i][1] == 9:
                    self.add_triangle(i, nodeids, element_nodes, 2)
                elif self.element_infos[i][1] == 3:
                    for j in range(4):
                        self.quads[self.num_quads][j] = nodeids[element_nodes[j]]
                    self.quads[self.num_quads][4] = self.element_infos[i][tagidx]
                    self.num_quads = self.num_quads + 1

                # TODO tetras/hexes/prisms/pyramids

        return True

    def add_line(self, i, nodeids, element_nodes, order):
        self.lines[self.num_lines][0] = nodeids[element_nodes[0]]
        self.lines[self.num_lines][1] = nodeids[element_nodes[1]]
        self.lines[self.num_lines][2] = self.element_infos[i][3]
        self.num_lines = self.num_lines + 1
        if order == 2:
            self.lines3[self.num_lines3][0] = nodeids[element_nodes[0]]
            self.lines3[self.num_lines3][1] = nodeids[element_nodes[1]]
            self.lines3[self.num_lines3][2] = nodeids[element_nodes[2]]
            self.lines3[self.num_lines3][3] = self.element_infos[i][3]
            self.num_lines3 = self.num_lines3 + 1

    def add_triangle(self, i, nodeids, element_nodes, order):
        for j in range(3):
            self.tris[self.num_tris][j] = nodeids[element_nodes[j]]
        self.tris[self.num_tris][3] = self.element_infos[i][3]
        self.num_tris = self.num_tris + 1
        if order == 2:
            for j in range(6):
                self.tris6[self.num_tris6][j] = nodeids[element_nodes[j]]
            self.tris6[self.num_tris6][6] = self.element_infos[i][3]
            self.num_tris6 = self.num_tris6 + 1
