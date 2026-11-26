#ifndef CVFEM_ELEMENT_TRAITS_H
#define CVFEM_ELEMENT_TRAITS_H

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/NgpTypes.hpp>

#include <cstddef>

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

} // namespace detail

struct Tet4SCVElement
{
  static constexpr int nodesPerElement = 4;
  static constexpr int numScvIp        = 4;
  static constexpr int spatialDim      = 3;

  using ExecSpace =
    stk::ngp::ExecSpace;

  using ScratchSpace =
    typename ExecSpace::scratch_memory_space;

  // Element coordinates: [node][dimension]
  using CoordinatesView =
    Kokkos::View<
      double**,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;
  
  // Shape-function derivatives: [SCV integration point][node][dimension]
  using DndxView =
    Kokkos::View<
      double***,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;

  // SCV volumes: [SCV integration point]
  using ScvVolumeView =
    Kokkos::View<
      double*,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;

  // Scratch requirements are exposed independently.
  // An algorithm only allocates what it actually uses.
  static constexpr std::size_t
  coordinates_scratch_size()
  {
    return sizeof(double)
      * nodesPerElement
      * spatialDim;
  }

  static constexpr std::size_t
  dndx_scratch_size()
  {
    return sizeof(double)
      * numScvIp
      * nodesPerElement
      * spatialDim;
  }


  static constexpr std::size_t
  scv_volume_scratch_size()
  {
    return sizeof(double)
      * numScvIp;
  }

  // SCV element operations.
  KOKKOS_FUNCTION
  static void dndx(
    const CoordinatesView& coordinates,
    const DndxView& dndx)
  {
    // Shape functions of the linear tet in reference coordinates:
    //   N0 = 1 - xi - eta - zeta,  N1 = xi,  N2 = eta,  N3 = zeta.
    //
    // The Jacobian is J = [e1 e2 e3] (edge vectors as columns), with
    //   e_k = x_k - x_0.
    // J and the reference-space derivatives are constant over the element, so
    // grad(N_i) has the same value at every SCV integration point
    // (ip_location). It is computed once and stored for each ip.
    //
    // The rows of inv(J) are (e2 x e3)/det, (e3 x e1)/det and (e1 x e2)/det,
    // which are grad(N_1), grad(N_2) and grad(N_3). grad(N_0) follows from
    // the partition of unity: sum_i grad(N_i) = 0.
    double e[3][spatialDim];

    for (int k = 0; k < 3; ++k) {
      for (int dim = 0; dim < spatialDim; ++dim) {
        e[k][dim] = coordinates(k + 1, dim) - coordinates(0, dim);
      }
    }

    double grad[nodesPerElement][spatialDim];

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

    for (int dim = 0; dim < spatialDim; ++dim) {

      grad[1][dim] *= invDet;
      grad[2][dim] *= invDet;
      grad[3][dim] *= invDet;

      grad[0][dim] = -(grad[1][dim] + grad[2][dim] + grad[3][dim]);
    }

    for (int ip = 0; ip < numScvIp; ++ip) {
      for (int node = 0; node < nodesPerElement; ++node) {
        for (int dim = 0; dim < spatialDim; ++dim) {
          dndx(ip, node, dim) = grad[node][dim];
        }
      }
    }
  }

  KOKKOS_FUNCTION
  static void determinant(
    const CoordinatesView& coordinates,
    const ScvVolumeView& scvVolume)
  {
    // Each node's SCV is ONE hexahedron whose eight corners are
    //
    //   the node itself                         (N)
    //   the midpoints of the 3 edges at N       (m_a, m_b, m_c)
    //   the centroids of the 3 faces at N       (f_ab, f_ac, f_bc)
    //   the element centroid                    (C)
    //
    // where a, b, c are the node's three neighbors in the element.
    //
    // The hex is a topological cube with reference axes
    //   xi   -> toward neighbor a
    //   eta  -> toward neighbor b
    //   zeta -> toward neighbor c
    // and is stored in the standard Hex8 (Exodus) corner order:
    //
    //   hex corner : (xi,eta,zeta) : point
    //        0     :  (0,0,0)      : N
    //        1     :  (1,0,0)      : m_a
    //        2     :  (1,1,0)      : f_ab
    //        3     :  (0,1,0)      : m_b
    //        4     :  (0,0,1)      : m_c
    //        5     :  (1,0,1)      : f_ac
    //        6     :  (1,1,1)      : C
    //        7     :  (0,1,1)      : f_bc
    //
    // Orientation: (a,b,c) is chosen so that (N,a,b,c) is an EVEN
    // permutation of (0,1,2,3). For a positively oriented tet this makes
    // every SCV hex right-handed, i.e. det(J) > 0. An inverted element
    // therefore yields NEGATIVE volumes.
    const int nbr[4][3] = {
      {1, 2, 3},   // node 0
      {0, 3, 2},   // node 1
      {0, 1, 3},   // node 2
      {2, 1, 0}};  // node 3

    // scvVolume is indexed by integration point. The SCV for integration
    // point `ip` is the one belonging to node ip_node(ip).
    for (int ip = 0; ip < numScvIp; ++ip) {

      const int n = ip_node(ip);

      const int a = nbr[n][0];
      const int b = nbr[n][1];
      const int c = nbr[n][2];

      double hex[8][3];

      for (int d = 0; d < spatialDim; ++d) {

        const double xn = coordinates(n, d);
        const double xa = coordinates(a, d);
        const double xb = coordinates(b, d);
        const double xc = coordinates(c, d);

        hex[0][d] = xn;
        hex[1][d] = 0.5 * (xn + xa);
        hex[2][d] = (xn + xa + xb) / 3.0;
        hex[3][d] = 0.5 * (xn + xb);
        hex[4][d] = 0.5 * (xn + xc);
        hex[5][d] = (xn + xa + xc) / 3.0;
        hex[6][d] = 0.25 * (xn + xa + xb + xc);
        hex[7][d] = (xn + xb + xc) / 3.0;
      }

      scvVolume(ip) = detail::hex8_volume(hex);
    }
  }

  // ipNodeMap: maps each SCV integration point to the element node that
  // owns it, i.e. ipNodeMap[ip] = node.
  //
  // The table is a local array, so this function is callable from device code.
  KOKKOS_FUNCTION
  static int ip_node(
    const int ip)
  {
    const int ipNodeMap[numScvIp] = {0, 1, 2, 3};
    return ipNodeMap[ip];
  }

  // Standard SCV integration-point locations, in the reference tet
  // (node 0 at the origin, node i on axis i): ip_location(ip, dim) is the
  // dim-th reference coordinate of integration point ip.
  //
  //   ip 0 : SCV of node 0      ip 1 : SCV of node 1
  //   ip 2 : SCV of node 2      ip 3 : SCV of node 3
  //
  // Same local-array pattern as ip_node() so it is device-callable.
  KOKKOS_FUNCTION
  static double ip_location(
    const int ip,
    const int dim)
  {
    const double seventeen96ths = 17.0 / 96.0;
    const double fourfive96ths  = 45.0 / 96.0;

    const double intgLoc[numScvIp][spatialDim] = {
      {seventeen96ths, seventeen96ths, seventeen96ths},  // vol 1
      {fourfive96ths,  seventeen96ths, seventeen96ths},  // vol 2
      {seventeen96ths, fourfive96ths,  seventeen96ths},  // vol 3
      {seventeen96ths, seventeen96ths, fourfive96ths}};  // vol 4

    return intgLoc[ip][dim];
  }
};

struct Tet4SCSElement
{
  static constexpr int nodesPerElement = 4;
  static constexpr int numScsIp        = 6;
  static constexpr int spatialDim      = 3;

  using ExecSpace =
    stk::ngp::ExecSpace;

  using ScratchSpace =
    typename ExecSpace::scratch_memory_space;

  // Element coordinates: [node][dimension]
  using CoordinatesView =
    Kokkos::View<
      double**,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;

  // Shape-function derivatives: [SCS integration point][node][dimension]
  using DndxView =
    Kokkos::View<
      double***,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;

  // SCS area vectors: [SCS integration point][dimension]
  using AreaVecView =
    Kokkos::View<
      double**,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;

  // Scratch requirements are exposed independently.
  // An algorithm only allocates what it actually uses.
  static constexpr std::size_t
  coordinates_scratch_size()
  {
    return sizeof(double)
      * nodesPerElement
      * spatialDim;
  }

  static constexpr std::size_t
  dndx_scratch_size()
  {
    return sizeof(double)
      * numScsIp
      * nodesPerElement
      * spatialDim;
  }

  static constexpr std::size_t
  area_vec_scratch_size()
  {
    return sizeof(double)
      * numScsIp
      * spatialDim;
  }

  // SCS element operations.
  KOKKOS_FUNCTION
  static void determinant(
    const CoordinatesView& coordinates,
    const AreaVecView& areav)
  {
    // Tet4 implementation TBD.
  }

  KOKKOS_FUNCTION
  static void dndx(
    const CoordinatesView& coordinates,
    const DndxView& dndx)
  {
    // Tet4 implementation TBD.
  }

  // lrscv: the left and right element node of each SCS surface, i.e.
  // lrscv(ip, 0) is the left node and lrscv(ip, 1) the right node of the
  // element edge that SCS integration point ip lies on.
  //
  //   ip 0 : nodes 0 -> 1      ip 3 : nodes 0 -> 3
  //   ip 1 : nodes 1 -> 2      ip 4 : nodes 1 -> 3
  //   ip 2 : nodes 0 -> 2      ip 5 : nodes 2 -> 3
  //
  // The table is a local array, so this function is callable from device code.
  KOKKOS_FUNCTION
  static int lrscv(
    const int ip,
    const int scs)
  {
    const int lrscvMap[numScsIp][2] = {
      {0, 1},
      {1, 2},
      {0, 2},
      {0, 3},
      {1, 3},
      {2, 3}};

    return lrscvMap[ip][scs];
  }

  // Standard SCS integration-point locations, in the reference tet
  // (node 0 at the origin, node i on axis i): ip_location(ip, dim) is the
  // dim-th reference coordinate of integration point ip. Each point is the
  // area centroid of its SCS surface.
  //
  // Same local-array pattern as lrscv() so it is device-callable.
  KOKKOS_FUNCTION
  static double ip_location(
    const int ip,
    const int dim)
  {
    const double thirteen36ths = 13.0 / 36.0;
    const double five36ths     =  5.0 / 36.0;

    const double intgLoc[numScsIp][spatialDim] = {
      {thirteen36ths, five36ths,     five36ths},      // surf 1    0->1
      {thirteen36ths, thirteen36ths, five36ths},      // surf 2    1->2
      {five36ths,     thirteen36ths, five36ths},      // surf 3    0->2
      {five36ths,     five36ths,     thirteen36ths},  // surf 4    0->3
      {thirteen36ths, five36ths,     thirteen36ths},  // surf 5    1->3
      {five36ths,     thirteen36ths, thirteen36ths}}; // surf 6    2->3

    return intgLoc[ip][dim];
  }
};

} // namespace cvfem

#endif
