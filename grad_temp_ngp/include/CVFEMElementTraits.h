#ifndef CVFEM_ELEMENT_TRAITS_H
#define CVFEM_ELEMENT_TRAITS_H

#include <Kokkos_Core.hpp>

#include <stk_mesh/base/NgpTypes.hpp>

#include <cstddef>

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


  //
  // Element coordinates:
  //
  //   [node][dimension]
  //
  using CoordinatesView =
    Kokkos::View<
      double**,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;


  //
  // Shape-function derivatives:
  //
  //   [SCV integration point][node][dimension]
  //
  using DndxView =
    Kokkos::View<
      double***,
      Kokkos::LayoutRight,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;


  //
  // SCV volumes:
  //
  //   [SCV integration point]
  //
  using ScvVolumeView =
    Kokkos::View<
      double*,
      ScratchSpace,
      Kokkos::MemoryTraits<
        Kokkos::Unmanaged>>;


  //
  // Scratch requirements are exposed independently.
  // An algorithm only allocates what it actually uses.
  //
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


  //
  // SCV element operations.
  //
  KOKKOS_FUNCTION
  static void compute_dndx(
    const CoordinatesView& coordinates,
    const DndxView& dndx)
  {
    // Tet4 implementation.
  }


  KOKKOS_FUNCTION
  static void compute_scv_volume(
    const CoordinatesView& coordinates,
    const ScvVolumeView& scvVolume)
  {
    // Tet4 implementation.
  }


  //
  // Mapping of SCV integration point to its associated
  // element node.
  //
  KOKKOS_FUNCTION
  static int ip_node(
    const int ip)
  {
    return ip;
  }
};

} // namespace cvfem

#endif
