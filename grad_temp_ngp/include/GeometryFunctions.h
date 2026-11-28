#ifndef CVFEM_GEOMETRY_FUNCTIONS_H
#define CVFEM_GEOMETRY_FUNCTIONS_H

#include <Kokkos_Core.hpp>

// Geometry helpers shared by the CVFEM element classes.

namespace cvfem
{

namespace detail
{

// Volume of a trilinear Hex8 patch: the sum over a 2x2x2 Gauss rule of
// det(J), with J_{rc} = sum_n x_{n,r} dN_n/dxi_c.
//
// `x` holds the eight corners in standard Hex8 (Exodus) order. The rule is
// exact for trilinear geometry: det(J) is at most quadratic in each
// reference coordinate, and 2-point Gauss integrates up to cubic. All eight
// Gauss weights are 1 on [-1,1]^3, so no weight appears below.
KOKKOS_INLINE_FUNCTION
double hex8_volume(const double x[8][3])
{
  // Reference-coordinate signs of the eight corners, [-1,1]^3 convention.
  const double sgn[8][3] = {
    {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
    {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}};

  // 2-point Gauss abscissa on [-1,1]: +/- 1/sqrt(3).
  const double g = Kokkos::sqrt(1.0 / 3.0);

  double volume = 0.0;

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {

        const double xi[3] = {i ? g : -g,
                              j ? g : -g,
                              k ? g : -g};

        // J[physical row][reference column]
        double J[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

        for (int n = 0; n < 8; ++n) {

          const double f0 = 1.0 + sgn[n][0] * xi[0];
          const double f1 = 1.0 + sgn[n][1] * xi[1];
          const double f2 = 1.0 + sgn[n][2] * xi[2];

          // N_n = (1/8) f0 f1 f2, so dN_n/dxi_c is:
          const double dN[3] = {0.125 * sgn[n][0] * f1 * f2,
                                0.125 * sgn[n][1] * f0 * f2,
                                0.125 * sgn[n][2] * f0 * f1};

          for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
              J[r][c] += x[n][r] * dN[c];
            }
          }
        }

        volume +=
            J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
          - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
          + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
      }
    }
  }

  return volume;
}

// Area vector of a bilinear quad: the integral over [-1,1]^2 of the surface
// Jacobian vector dx/ds x dx/dt.
//
// For a bilinear quad dx/ds = a + c t and dx/dt = b + c s, so the integrand
// is a x b + s (a x c) + t (c x b), which is linear in (s,t). A single point
// at the quad center (s = t = 0), weighted by the reference area 4,
// integrates it exactly. The result equals the vector area of the
// triangulated quad.
//
// `x` holds the four corners in order around the quad. The area vector
// follows the right-hand rule on that order: it points toward the viewer who
// sees the corners 0 -> 1 -> 2 -> 3 run counterclockwise.
KOKKOS_INLINE_FUNCTION
void quad4_area_vector(const double x[4][3], double areav[3])
{
  // dx/ds and dx/dt at the quad center. Corner signs in (s,t) are
  //   0: (-1,-1)  1: (+1,-1)  2: (+1,+1)  3: (-1,+1)
  double dxds[3];
  double dxdt[3];

  for (int dim = 0; dim < 3; ++dim) {
    dxds[dim] = 0.25 * (-x[0][dim] + x[1][dim] + x[2][dim] - x[3][dim]);
    dxdt[dim] = 0.25 * (-x[0][dim] - x[1][dim] + x[2][dim] + x[3][dim]);
  }

  // Single-point weight: the reference quad area, 2 x 2 = 4.
  areav[0] = 4.0 * (dxds[1] * dxdt[2] - dxds[2] * dxdt[1]);
  areav[1] = 4.0 * (dxds[2] * dxdt[0] - dxds[0] * dxdt[2]);
  areav[2] = 4.0 * (dxds[0] * dxdt[1] - dxds[1] * dxdt[0]);
}

// Gradients of the linear tet shape functions with respect to the physical
// coordinates: grad[i][dim] = dN_i / dx_dim. `coordinates` is indexed as
// coordinates(node, dim).
//
// Shape functions in reference coordinates:
//   N0 = 1 - xi - eta - zeta,  N1 = xi,  N2 = eta,  N3 = zeta.
//
// The Jacobian is J = [e1 e2 e3] (edge vectors as columns), with
//   e_k = x_k - x_0.
// J and the reference-space derivatives are constant over the element, so
// the physical gradients are constant too.
//
// The rows of inv(J) are (e2 x e3)/det, (e3 x e1)/det and (e1 x e2)/det,
// which are grad(N_1), grad(N_2) and grad(N_3). grad(N_0) follows from
// the partition of unity: sum_i grad(N_i) = 0.
KOKKOS_INLINE_FUNCTION
void tet4_shape_function_gradients(
  const auto& coordinates,
  double grad[4][3])
{
  double e[3][3];

  for (int k = 0; k < 3; ++k) {
    for (int dim = 0; dim < 3; ++dim) {
      e[k][dim] = coordinates(k + 1, dim) - coordinates(0, dim);
    }
  }

  for (int k = 0; k < 3; ++k) {
    const double* a = e[(k + 1) % 3];
    const double* b = e[(k + 2) % 3];

    grad[k + 1][0] = a[1] * b[2] - a[2] * b[1];
    grad[k + 1][1] = a[2] * b[0] - a[0] * b[2];
    grad[k + 1][2] = a[0] * b[1] - a[1] * b[0];
  }

  // det(J) = e1 . (e2 x e3)
  const double det =
      e[0][0] * grad[1][0]
    + e[0][1] * grad[1][1]
    + e[0][2] * grad[1][2];

  const double invDet = 1.0 / det;

  for (int dim = 0; dim < 3; ++dim) {

    grad[1][dim] *= invDet;
    grad[2][dim] *= invDet;
    grad[3][dim] *= invDet;

    grad[0][dim] = -(grad[1][dim] + grad[2][dim] + grad[3][dim]);
  }
}

// Shape functions of the linear tet at the reference point (xi, eta, zeta):
//   N0 = 1 - xi - eta - zeta,  N1 = xi,  N2 = eta,  N3 = zeta.
KOKKOS_INLINE_FUNCTION
void tet4_shape_functions(
  const double xi,
  const double eta,
  const double zeta,
  double shape[4])
{
  shape[0] = 1.0 - xi - eta - zeta;
  shape[1] = xi;
  shape[2] = eta;
  shape[3] = zeta;
}

} // namespace detail

} // namespace cvfem

#endif
