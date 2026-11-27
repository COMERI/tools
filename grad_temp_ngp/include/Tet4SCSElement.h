#ifndef TET4_SCS_ELEMENT_H
#define TET4_SCS_ELEMENT_H

#include "GeometryFunctions.h"

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/NgpTypes.hpp>

#include <cstddef>

// Tet4 sub-control-surface (SCS) element.

namespace cvfem
{

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
    // Each SCS surface is a quad whose corners are one edge midpoint, the
    // centroids of the two element faces that share that edge, and the
    // element centroid. The points are numbered as follows:
    //
    //    0-3  element nodes
    //    4    midpoint of edge 0-1     5    midpoint of edge 1-2
    //    6    midpoint of edge 2-0     7    centroid of face 0-1-2
    //    8    midpoint of edge 2-3     9    midpoint of edge 3-1
    //   10    centroid of face 1-2-3
    //   11    midpoint of edge 0-3     12   centroid of face 0-2-3
    //   13    centroid of face 0-1-3
    //   14    element centroid
    double pt[15][spatialDim];

    for (int dim = 0; dim < spatialDim; ++dim) {

      const double x0 = coordinates(0, dim);
      const double x1 = coordinates(1, dim);
      const double x2 = coordinates(2, dim);
      const double x3 = coordinates(3, dim);

      pt[0][dim] = x0;
      pt[1][dim] = x1;
      pt[2][dim] = x2;
      pt[3][dim] = x3;

      pt[4][dim]  = 0.5 * (x0 + x1);
      pt[5][dim]  = 0.5 * (x1 + x2);
      pt[6][dim]  = 0.5 * (x2 + x0);
      pt[7][dim]  = (x0 + x1 + x2) / 3.0;

      pt[8][dim]  = 0.5 * (x2 + x3);
      pt[9][dim]  = 0.5 * (x3 + x1);
      pt[10][dim] = (x1 + x2 + x3) / 3.0;

      pt[11][dim] = 0.5 * (x0 + x3);
      pt[12][dim] = (x0 + x2 + x3) / 3.0;

      pt[13][dim] = (x0 + x1 + x3) / 3.0;

      pt[14][dim] = 0.25 * (x0 + x1 + x2 + x3);
    }

    // Quad corners of SCS integration point ip, in the order that makes its
    // area vector point from lrscv(ip, 0) toward lrscv(ip, 1).
    const int quadPoints[numScsIp][4] = {
      { 4,  7, 14, 13},
      { 7, 14, 10,  5},
      { 6, 12, 14,  7},
      {11, 13, 14, 12},
      {13,  9, 10, 14},
      {10,  8, 12, 14}};

    for (int ip = 0; ip < numScsIp; ++ip) {

      double quad[4][spatialDim];

      for (int corner = 0; corner < 4; ++corner) {
        for (int dim = 0; dim < spatialDim; ++dim) {
          quad[corner][dim] = pt[quadPoints[ip][corner]][dim];
        }
      }

      double av[spatialDim];
      detail::quad4_area_vector(quad, av);

      for (int dim = 0; dim < spatialDim; ++dim) {
        areav(ip, dim) = av[dim];
      }
    }
  }

  KOKKOS_FUNCTION
  static void grad_op(
    const CoordinatesView& coordinates,
    const DndxView& dndx)
  {
    // The gradients of the linear tet shape functions are constant over the
    // element, so they are computed once and stored for each SCS
    // integration point.
    double grad[nodesPerElement][spatialDim];

    detail::tet4_shape_function_gradients(coordinates, grad);

    for (int ip = 0; ip < numScsIp; ++ip) {
      for (int node = 0; node < nodesPerElement; ++node) {
        for (int dim = 0; dim < spatialDim; ++dim) {
          dndx(ip, node, dim) = grad[node][dim];
        }
      }
    }
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
  //
  KOKKOS_FUNCTION
  static int lrscv(
    const int ip,
    const int side)
  {
    const int lrscvMap[numScsIp][2] = {
      {0, 1},
      {1, 2},
      {0, 2},
      {0, 3},
      {1, 3},
      {2, 3}};

    return lrscvMap[ip][side];
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
