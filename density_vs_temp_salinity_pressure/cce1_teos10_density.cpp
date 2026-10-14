// ============================================================
// cce1_teos10_density.cpp
//
// Calculate seawater density for the CCE1 CTD dataset using
// the TEOS-10 Gibbs Seawater (GSW-C) library.
//
// The program calculates:
//
//   1. Practical Salinity (SP)
//      - Directly from the OceanSITES PSAL measurements.
//
//   2. Absolute Salinity (SA)
//      - Calculated from SP using TEOS-10.
//      - Corrected using pressure, longitude and latitude.
//
//   3. Conservative Temperature (CT)
//      - Calculated from in-situ temperature.
//
//   4. Seawater density
//      - Calculated from SA, CT and pressure.
//
//   5. A perturbed case
//      - Temperature is increased by perturbedTemp.
//      - Salinity, depth and location remain unchanged.
//
// Output:
//   cce1_teos10_density.csv
//
// Requires the TEOS-10 GSW-C library.
// ============================================================

#include <iostream>
#include <iomanip>
#include <fstream>

#include "gswteos-10.h"

using namespace std;


// ============================================================
// USER SETTINGS
// ============================================================

// Temperature perturbation in degrees C.
//
// Change this value to test different warming scenarios.
//
// Examples:
//     0.5  -> 0.5 degree C warming
//     1.0  -> 1.0 degree C warming
//     2.0  -> 2.0 degree C warming
//
const double perturbedTemp = 1.0;


// ============================================================
// CCE1 STATION LOCATION
// ============================================================
//
// These are the latitude and longitude from the OceanSITES
// CCE1 CTD dataset.
//
// Longitude follows the GSW convention:
//     East  = positive
//     West  = negative
//
// Therefore -122.5158 means 122.5158 degrees West.
//

const double latitude  = 33.5283;
const double longitude = -122.5158;


// ============================================================
// DEPTHS
// ============================================================
//
// Depth is positive downward in the OceanSITES dataset.
//
// GSW uses z < 0 below the surface, so the program converts:
//
//     depth = +10 m
//
// into:
//
//     z = -10 m
//

const int NDEPTH = 13;

const double depth[NDEPTH] =
{
    10.0,
    20.0,
    31.0,
    41.0,
    61.0,
    78.0,
    145.0,
    302.0,
    499.0,
    754.0,
    1001.0,
    2213.0,
    3873.0
};


// ============================================================
// OBSERVATION TIMES
// ============================================================
//
// These are the actual TIME values supplied in the dataset.
//
// Units are days since 1950-01-01.
//

const int NTIME = 13;

const double observationTime[NTIME] =
{
    23284.000011574128,
    23284.020844907383,
    23284.041678240756,
    23284.062511574128,
    23284.083344907383,
    23284.104178240756,
    23284.125011574128,
    23284.145844907383,
    23284.166678240756,
    23284.187511574128,
    23284.208344907383,
    23284.229178240756,
    23284.250011574128
};


// ============================================================
// TEMPERATURE DATA
// ============================================================
//
// temperature[time][depth]
//
// Units: degrees C
//

const double temperature[NTIME][NDEPTH] =
{
    {
        17.97639, 17.7704, 17.70251, 17.452507,
        12.388808, 10.733809, 8.9919, 7.1544,
        5.79631, 4.76891, 3.9744108, 1.9548115, 1.5085092
    },

    {
        18.046692, 17.9764, 17.61471, 17.295609,
        11.872808, 10.77281, 9.0633, 7.2045,
        5.67311, 4.79791, 4.0479107, 1.9595115, 1.5099092
    },

    {
        18.330791, 18.0214, 17.834211, 17.118008,
        12.134008, 10.76321, 8.9452, 7.1199,
        5.69861, 4.76511, 3.981011, 1.9745116, 1.5096092
    },

    {
        18.345392, 18.0794, 17.730711, 17.084208,
        12.478309, 11.15221, 8.9852, 7.1183,
        6.0322104, 4.71461, 3.994711, 1.9723116, 1.5107093
    },

    {
        18.33509, 18.0674, 17.551811, 15.442108,
        11.942509, 10.70771, 9.1106, 7.1269,
        5.7962103, 4.7126102, 3.973311, 1.9703118, 1.5108094
    },

    {
        18.191092, 18.0774, 17.568111, 16.588108,
        12.327509, 11.28121, 9.0473, 7.102,
        5.6425104, 4.70241, 3.941011, 1.9639118, 1.5095094
    },

    {
        17.882292, 17.8504, 17.843811, 16.609608,
        12.297309, 11.182811, 9.1084, 7.0994,
        5.8069105, 4.6955104, 3.979711, 1.9588119, 1.5092095
    },

    {
        18.189392, 18.1104, 16.576912, 13.850808,
        11.642909, 10.58671, 8.8612, 7.0957,
        5.6328106, 4.7016106, 3.977811, 1.955012, 1.5088096
    },

    {
        18.272892, 18.1284, 16.27531, 13.960508,
        11.598908, 10.57551, 8.9564, 7.0984,
        5.5830107, 4.7088103, 3.9375112, 1.946012, 1.5084096
    },

    {
        18.347792, 18.2284, 17.930212, 14.430308,
        11.619709, 10.54341, 8.971, 7.083,
        5.5836105, 4.7070107, 3.9304113, 1.9476122, 1.5088097
    },

    {
        18.39199, 18.3124, 18.250412, 14.623908,
        12.246909, 10.719411, 8.9972, 7.1252,
        5.5765104, 4.6987104, 3.9247115, 1.9501122, 1.5072098
    },

    {
        18.42139, 18.3574, 18.31961, 13.993208,
        11.582309, 10.516211, 8.966, 7.0935,
        5.635811, 4.6925106, 3.9146116, 1.9717124, 1.5060098
    },

    {
        18.46759, 18.4144, 18.370811, 14.682308,
        11.644309, 10.5618105, 8.9918, 7.1027,
        5.587411, 4.701111, 3.9202116, 1.9669124, 1.5063099
    }
};


// ============================================================
// PRACTICAL SALINITY DATA
// ============================================================
//
// This is the PSAL variable from the OceanSITES dataset.
//
// Practical Salinity is conventionally denoted:
//
//     SP
//
// It is NOT Absolute Salinity.
//
// Units are Practical Salinity (dimensionless; often reported
// as PSU).
//
// The values below are used as the input to the TEOS-10
// SP -> SA conversion.
//

const double practicalSalinity[NTIME][NDEPTH] =
{
    {
        33.471275, 33.47664, 33.510986, 33.57857,
        33.024246, 32.990643, 33.6991, 34.01799,
        34.182335, 34.36492, 34.455486, 34.610863, 34.67621
    },

    {
        33.462276, 33.465317, 33.524895, 33.583744,
        32.94062, 32.994354, 33.660233, 34.018364,
        34.17639, 34.36483, 34.450447, 34.610397, 34.675575
    },

    {
        33.40731, 33.46035, 33.44508, 33.570103,
        32.977882, 32.99312, 33.722713, 34.031734,
        34.187965, 34.367554, 34.457916, 34.609875, 34.67578
    },

    {
        33.407772, 33.45093, 33.512688, 33.54809,
        32.995167, 32.93132, 33.702003, 34.0332,
        34.19695, 34.379566, 34.456257, 34.609398, 34.67545
    },

    {
        33.40177, 33.45372, 33.564617, 33.228813,
        32.90935, 32.98026, 33.64415, 34.02321,
        34.194954, 34.37734, 34.455925, 34.60846, 34.675724
    },

    {
        33.407284, 33.445625, 33.43112, 33.396606,
        32.987495, 32.952003, 33.67319, 34.044113,
        34.18693, 34.378452, 34.46022, 34.610493, 34.675907
    },

    {
        33.375935, 33.36163, 33.42733, 33.34872,
        32.97668, 32.947212, 33.63054, 34.045578,
        34.20232, 34.38208, 34.458447, 34.612247, 34.675552
    },

    {
        33.436947, 33.449203, 33.1764, 32.97895,
        32.959606, 33.007446, 33.771267, 34.044964,
        34.183826, 34.379276, 34.457718, 34.61102, 34.675983
    },

    {
        33.40679, 33.43018, 33.197388, 32.97576,
        32.965984, 33.014194, 33.72354, 34.043068,
        34.187607, 34.381596, 34.458088, 34.612755, 34.67612
    },

    {
        33.395695, 33.41609, 33.392025, 32.973167,
        32.936363, 33.02286, 33.713634, 34.042717,
        34.188248, 34.378815, 34.458763, 34.612526, 34.675865
    },

    {
        33.38461, 33.397762, 33.413715, 32.991882,
        33.10367, 32.98587, 33.69703, 34.03318,
        34.18613, 34.385273, 34.45692, 34.611088, 34.676327
    },

    {
        33.385143, 33.40561, 33.420242, 32.96759,
        32.951164, 33.04332, 33.718445, 34.041664,
        34.18544, 34.385517, 34.46261, 34.608932, 34.67651
    },

    {
        33.375095, 33.391827, 33.430126, 32.96316,
        32.951965, 33.015995, 33.70179, 34.04102,
        34.194607, 34.37979, 34.462082, 34.609875, 34.676083
    }
};


// ============================================================
// MAIN PROGRAM
// ============================================================

int main()
{
    // --------------------------------------------------------
    // Open CSV output file.
    //
    // There are:
    //
    //     13 times x 13 depths = 169 observations
    //
    // so the output will contain 169 data rows.
    // --------------------------------------------------------

    ofstream output("cce1_teos10_density.csv");

    if (!output)
    {
        cerr << "ERROR: Could not open output file.\n";
        return 1;
    }


    // --------------------------------------------------------
    // CSV header.
    //
    // SP = Practical Salinity
    // SA = Absolute Salinity
    //
    // Keeping both in the output makes it possible to verify
    // the TEOS-10 salinity conversion independently.
    // --------------------------------------------------------

    output
        << "time,"
        << "depth_m,"
        << "temperature_C,"
        << "perturbed_temperature_C,"
        << "SP_practical,"
        << "SA_absolute_gkg,"
        << "pressure_dbar,"
        << "CT_C,"
        << "perturbed_CT_C,"
        << "density_kg_m3,"
        << "perturbed_density_kg_m3,"
        << "density_difference_kg_m3\n";


    // --------------------------------------------------------
    // Loop over every observation time.
    // --------------------------------------------------------

    for (int it = 0; it < NTIME; ++it)
    {

        // ----------------------------------------------------
        // Loop over every measured depth.
        // ----------------------------------------------------

        for (int iz = 0; iz < NDEPTH; ++iz)
        {

            // =================================================
            // ORIGINAL INPUT DATA
            // =================================================

            // Measured in-situ temperature from the CTD.
            double t = temperature[it][iz];

            // Practical Salinity from the CTD PSAL variable.
            //
            // This is SP, NOT Absolute Salinity.
            //
            double SP = practicalSalinity[it][iz];

            // Positive-downward depth from the dataset.
            double z = depth[iz];

            // =================================================
            // PRESSURE
            // =================================================
            //
            // GSW expects depth z to be negative below the
            // surface.
            //
            // Example:
            //
            //     dataset depth = +145 m
            //
            // becomes:
            //
            //     GSW z = -145 m
            //
            // gsw_p_from_z() returns sea pressure in dbar.
            //
            // Latitude is included because the conversion from
            // depth to pressure depends slightly on latitude.
            //

	    double pressure = gsw_p_from_z(
               -z,        // depth in meters, negative below sea surface
               latitude,  // station latitude
               0.0,       // dynamic height anomaly at sea surface
               0.0        // sea-surface geopotential
            );

            // =================================================
            // PRACTICAL SALINITY -> ABSOLUTE SALINITY
            // =================================================
            //
            // This is the important TEOS-10 salinity conversion.
            //
            // INPUT:
            //
            //     SP         Practical Salinity
            //     pressure   pressure in dbar
            //     longitude  station longitude
            //     latitude   station latitude
            //
            // OUTPUT:
            //
            //     SA         Absolute Salinity in g/kg
            //
            // Absolute Salinity therefore incorporates the
            // geographical salinity anomaly associated with the
            // actual location of the CCE1 station.
            //

            double SA = gsw_sa_from_sp(
                SP,
                pressure,
                longitude,
                latitude
            );


            // =================================================
            // IN-SITU TEMPERATURE -> CONSERVATIVE TEMPERATURE
            // =================================================
            //
            // TEOS-10 uses Conservative Temperature (CT) for
            // the thermodynamic equation of state.
            //
            // CT is calculated from:
            //
            //     Absolute Salinity
            //     in-situ temperature
            //     pressure
            //
            double CT = gsw_ct_from_t(
                SA,
                t,
                pressure
            );


            // =================================================
            // ORIGINAL SEAWATER DENSITY
            // =================================================
            //
            // TEOS-10 density:
            //
            //     rho = rho(SA, CT, pressure)
            //
            // Result is kg/m^3.
            //

            double rho = gsw_rho(
                SA,
                CT,
                pressure
            );


            // =================================================
            // TEMPERATURE PERTURBATION
            // =================================================
            //
            // We change ONLY temperature.
            //
            // Practical Salinity remains unchanged.
            // Absolute Salinity remains unchanged.
            // Depth remains unchanged.
            // Location remains unchanged.
            //
            // Therefore:
            //
            //     perturbedT = originalT + perturbedTemp
            //

            double perturbedT = t + perturbedTemp;


            // =================================================
            // PERTURBED CONSERVATIVE TEMPERATURE
            // =================================================
            //
            // Convert the warmed in-situ temperature to CT.
            //
            // SA and pressure are unchanged.
            //

            double perturbedCT = gsw_ct_from_t(
                SA,
                perturbedT,
                pressure
            );


            // =================================================
            // PERTURBED SEAWATER DENSITY
            // =================================================
            //
            // Calculate density using the warmer temperature.
            //
            // Again:
            //
            //     SA       = unchanged
            //     pressure = unchanged
            //     CT       = changed
            //
            double perturbedRho = gsw_rho(
                SA,
                perturbedCT,
                pressure
            );


            // =================================================
            // DENSITY CHANGE
            // =================================================
            //
            // A negative value means that warming reduced
            // seawater density.
            //
            //     Delta rho =
            //       perturbed density - original density
            //

            double densityDifference =
                perturbedRho - rho;


            // =================================================
            // WRITE RESULT TO CSV
            // =================================================

            output
                << setprecision(12)
                << observationTime[it] << ","
                << z << ","
                << t << ","
                << perturbedT << ","
                << SP << ","
                << SA << ","
                << pressure << ","
                << CT << ","
                << perturbedCT << ","
                << rho << ","
                << perturbedRho << ","
                << densityDifference
                << "\n";
        }
    }


    // --------------------------------------------------------
    // Close output file.
    // --------------------------------------------------------

    output.close();


    // --------------------------------------------------------
    // Tell the user what was generated.
    // --------------------------------------------------------

    cout << "Calculation complete.\n";
    cout << "Output file: cce1_teos10_density.csv\n";
    cout << "Temperature perturbation: "
         << perturbedTemp
         << " deg C\n";

    cout << "Observations processed: "
         << NTIME * NDEPTH
         << "\n";

    return 0;
}

