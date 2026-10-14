/*------------------------------------------------------------------------*/
/*  Copyright 2025 COMERI.                                                */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Mare-Nalu     */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#ifndef WaterPropertyEvaluator_h
#define WaterPropertyEvaluator_h

#include <vector>

namespace sierra{
namespace nalu{

class WaterDensityTPropertyEvaluator
{
public:

  WaterDensityTPropertyEvaluator();
  virtual ~WaterDensityTPropertyEvaluator();
  
  double execute(
      double *indVarList);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
};

class WaterViscosityTPropertyEvaluator
{
public:

  WaterViscosityTPropertyEvaluator();
  virtual ~WaterViscosityTPropertyEvaluator();
  
  double execute(
      double *indVarList);
  
  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
};

class WaterSpecHeatTPropertyEvaluator
{
public:

  WaterSpecHeatTPropertyEvaluator();
  virtual ~WaterSpecHeatTPropertyEvaluator();
  
  double execute(
      double *indVarList);
  
  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
  const double ew_;
};

class WaterEnthalpyTPropertyEvaluator
{
public:

  WaterEnthalpyTPropertyEvaluator();
  virtual ~WaterEnthalpyTPropertyEvaluator();
  
  double execute(
    double *indVarList);

  double compute_h(
    const double T);

  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
  const double dw_;
  const double ew_;
  const double Tref_;
  const double hRef_;
};

class WaterThermalCondTPropertyEvaluator
{
public:

  WaterThermalCondTPropertyEvaluator();
  virtual ~WaterThermalCondTPropertyEvaluator();
  
  double execute(
      double *indVarList);
  
  // reference quantities
  const double aw_;
  const double bw_;
  const double cw_;
};

} // namespace nalu
} // namespace Sierra

#endif
