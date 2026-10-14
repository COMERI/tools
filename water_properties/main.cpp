#include <iostream>
#include <iomanip>
#include "WaterPropertyEvaluator.h"

int main()
{
    const double TempCool = 289.15; // K
    const double TempWarm = 296.15; // K
    const double U = 0.7;           // m/s
    const double L = 0.17;          // m

    sierra::nalu::WaterDensityTPropertyEvaluator rho;
    sierra::nalu::WaterViscosityTPropertyEvaluator mu;
    sierra::nalu::WaterSpecHeatTPropertyEvaluator cp;
    sierra::nalu::WaterEnthalpyTPropertyEvaluator h;
    sierra::nalu::WaterThermalCondTPropertyEvaluator k;

    // Cool and warm property values
    double rhoCool, rhoWarm;
    double muCool, muWarm;
    double cpCool, cpWarm;
    double hCool, hWarm;
    double kCool, kWarm;
    double ReCool, ReWarm;
    double PrCool, PrWarm;

    auto evaluate = [&](double T,
                        double& rhoVal, double& muVal,
                        double& cpVal, double& hVal,
                        double& kVal, double& Re,
                        double& Pr)
    {
        double indVarList[1] = {T};

        rhoVal = rho.execute(indVarList);
        muVal  = mu.execute(indVarList);
        cpVal  = cp.execute(indVarList);
        hVal   = h.execute(indVarList);
        kVal   = k.execute(indVarList);

        Re = rhoVal * U * L / muVal;
        Pr = muVal * cpVal / kVal;
    };

    evaluate(TempCool, rhoCool, muCool, cpCool, hCool,
             kCool, ReCool, PrCool);

    evaluate(TempWarm, rhoWarm, muWarm, cpWarm, hWarm,
             kWarm, ReWarm, PrWarm);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Property values\n"
              << "                Cool          Warm\n"
              << "Temperature  " << TempCool << "      " << TempWarm << " K\n"
              << "Density      " << rhoCool << "      " << rhoWarm << " kg/m^3\n"
              << "Viscosity    " << muCool  << "      " << muWarm  << " kg/(m s)\n"
              << "Specific heat" << cpCool  << "   " << cpWarm  << " J/(kg K)\n"
              << "Enthalpy     " << hCool   << "      " << hWarm   << " J/kg\n"
              << "Conductivity " << kCool   << "      " << kWarm   << " W/(m K)\n"
              << "Reynolds     " << ReCool  << "      " << ReWarm  << "\n"
              << "Prandtl      " << PrCool  << "      " << PrWarm  << "\n\n";

    std::cout << "Warm / Cool ratios\n"
              << "Density ratio      = " << rhoWarm / rhoCool << "\n"
              << "Viscosity ratio    = " << muWarm / muCool << "\n"
              << "Specific heat ratio= " << cpWarm / cpCool << "\n"
              << "Enthalpy ratio     = " << hWarm / hCool << "\n"
              << "Conductivity ratio = " << kWarm / kCool << "\n"
              << "Reynolds ratio     = " << ReWarm / ReCool << "\n"
              << "Prandtl ratio      = " << PrWarm / PrCool << "\n";

    std::cout << "\nPercentage changes (Warm relative to Cool)\n"
              << "Density change       = "
              << 100.0 * (rhoWarm / rhoCool - 1.0) << " %\n"
              << "Viscosity change     = "
              << 100.0 * (muWarm / muCool - 1.0) << " %\n"
              << "Specific heat change = "
              << 100.0 * (cpWarm / cpCool - 1.0) << " %\n"
              << "Enthalpy change      = "
              << 100.0 * (hWarm / hCool - 1.0) << " %\n"
              << "Conductivity change  = "
              << 100.0 * (kWarm / kCool - 1.0) << " %\n"
              << "Reynolds change      = "
              << 100.0 * (ReWarm / ReCool - 1.0) << " %\n"
              << "Prandtl change       = "
              << 100.0 * (PrWarm / PrCool - 1.0) << " %\n";

    return 0;
}
