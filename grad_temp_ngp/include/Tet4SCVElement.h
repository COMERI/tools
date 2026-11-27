#ifndef TET4_SCV_ELEMENT_H
#define TET4_SCV_ELEMENT_H

#include "GeometryFunctions.h"

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/NgpTypes.hpp>

#include <cstddef>

// Tet4 sub-control-volume (SCV) element.

namespace cvfem
{

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
  static void grad_op(
    const CoordinatesView& coordinates,
    const DndxView& dndx)
  {
    // The gradients of the linear tet shape functions are constant over the
    // element, so they are computed once and stored for each SCV
    // integration point.
    double grad[nodesPerElement][spatialDim];

    detail::tet4_shape_function_gradients(coordinates, grad);

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
  //
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

} // namespace cvfem

#endif
