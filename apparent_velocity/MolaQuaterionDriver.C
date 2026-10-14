#include <iostream>
#include <cmath>
#include <vector>

// Compute quaternion rotating nApp -> nE
// Quaternion convention: q = (w,x,y,z)
void compute_alignment_quaternion(
    const std::vector<double>& nApp,
    const std::vector<double>& nE,
    double& q0,
    double& q1,
    double& q2,
    double& q3)
{
  // Normalize nApp
  double denom = std::sqrt(
			   nApp[0]*nApp[0] +
			   nApp[1]*nApp[1] +
			   nApp[2]*nApp[2]);
  
  const double ax = nApp[0] / denom;
  const double ay = nApp[1] / denom;
  const double az = nApp[2] / denom;
  
  // Normalize nE
  denom = std::sqrt(
		    nE[0]*nE[0] +
		    nE[1]*nE[1] +
		    nE[2]*nE[2]);
  
  const double bx = nE[0] / denom;
  const double by = nE[1] / denom;
  const double bz = nE[2] / denom;
  
  
  // Dot product
  double D = ax*bx + ay*by + az*bz;
  
  // Clamp numerical errors
  D = std::max(-1.0, std::min(1.0, D));
  
  
  const double small = 1.0e-14;
  
  
  // Identity rotation
  if (D > 1.0 - small)
    {
      q0 = 1.0;
      q1 = 0.0;
      q2 = 0.0;
      q3 = 0.0;
      return;
    }
  
  
  // 180 degree rotation
  if (D < -1.0 + small)
    {
      // Pick a perpendicular axis
      double rx = 1.0;
      double ry = 0.0;
      double rz = 0.0;
      
      if (std::abs(ax) > 0.9)
        {
	  rx = 0.0;
	  ry = 1.0;
	  rz = 0.0;
        }
      
      // axis = nApp x reference
      double cx = ay*rz - az*ry;
      double cy = az*rx - ax*rz;
      double cz = ax*ry - ay*rx;
      
      double mag = std::sqrt(
			     cx*cx + cy*cy + cz*cz);
      
      cx /= mag;
      cy /= mag;
      cz /= mag;
      
      q0 = 0.0;
      q1 = cx;
      q2 = cy;
      q3 = cz;
      
      return;
    }
  
  
  // General shortest rotation
  // axis = nApp x nE
  const double cx = ay*bz - az*by;
  const double cy = az*bx - ax*bz;
  const double cz = ax*by - ay*bx;
  
  
  const double s = std::sqrt(2.0*(1.0 + D));
  
  q0 = 0.5*s;
  
  const double fac = 1.0/s;
  
  q1 = cx*fac;
  q2 = cy*fac;
  q3 = cz*fac;
}

int main() {

  const bool worryAboutOrientation = true;
  
  // save off unit vectors, desired n (normalize to be safe)
  std::vector<double> n(3);
  n[0] = 0.0;
  n[1] = -1.0/std::sqrt(2.0);
  n[2] = -1.0/std::sqrt(2.0);
  double denom = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
  const double nx = n[0]/denom;
  const double ny = n[1]/denom;
  const double nz = n[2]/denom;

  // provide initial direction n^o 
  std::vector<double> no(3);
  no[0] = 0.0;
  no[1] = -1.0;
  no[2] = 0.0;
  denom = std::sqrt(no[0]*no[0] + no[1]*no[1] + no[2]*no[2]);
  const double nox = no[0]/denom;
  const double noy = no[1]/denom;
  const double noz = no[2]/denom;

  // provide apparent velocity direction
  std::vector<double> na(3);
  na[0] = 0.0;
  na[1] = 0.842769;
  na[2] = 0.538275;

  // provide priciply aligned direction  
  std::vector<double> ne(3);
  ne[0] = 0.0;
  ne[1] = 1.0;
  ne[2] = 0.0;
  
  // provide initial roll angle
  const double theAngle = 15.0;
  const double theAngleInRad = theAngle*std::acos(-1.0)/180.0;    

  // construct the dot product
  double D = nox*nx + noy*ny + noz*nz;
 
  // error check (180 degree rotation singularity)
  const double small = 1.0e-14;
  if ( std::abs(1.0 + D) < small ) {
    std::cout << "WARNING: initial_rotation may be singular" << std::endl;
    D += small;
  }
  
  // normalization factor and sin/cos
  const double fac = 1.0/std::sqrt(2.0*(1.0+D));
  const double s = sin(theAngleInRad*0.5);
  const double c = cos(theAngleInRad*0.5);
  
  // form q (align and roll, q = qRoll * qAlign)
  double q0 = fac*c*(1+D);
  double q1 = fac*(c*(noy*nz - noz*ny) + s*(nox+nx));
  double q2 = fac*(c*(noz*nx - nox*nz) + s*(noy+ny));
  double q3 = fac*(c*(nox*ny - noy*nx) + s*(noz+nz));

  std::cout << "q0: " << q0 << std::endl;
  std::cout << "q1: " << q1 << std::endl;
  std::cout << "q2: " << q2 << std::endl;
  std::cout << "q3: " << q3 << std::endl;

  // now, handle possible CFD orientation...
  if ( worryAboutOrientation ) {
    // qFinal = qCFD * q
    
    double qCFD0, qCFD1, qCFD2, qCFD3;
    compute_alignment_quaternion(na, ne, qCFD0, qCFD1, qCFD2, qCFD3);

    std::cout << "qCFD0: " << qCFD0 << std::endl;
    std::cout << "qCFD1: " << qCFD1 << std::endl;
    std::cout << "qCFD2: " << qCFD2 << std::endl;  
    std::cout << "qCFD3: " << qCFD3 << std::endl;

    const double qFinal0 =
      qCFD0*q0
      - qCFD1*q1
      - qCFD2*q2
      - qCFD3*q3;
    
    const double qFinal1 =
      qCFD0*q1
      + qCFD1*q0
      + qCFD2*q3
      - qCFD3*q2;
    
    const double qFinal2 =
      qCFD0*q2
      - qCFD1*q3
      + qCFD2*q0
      + qCFD3*q1;
    
    const double qFinal3 =
      qCFD0*q3
      + qCFD1*q2
      - qCFD2*q1
      + qCFD3*q0;

    // provide to q
    q0 = qFinal0;
    q1 = qFinal1;
    q2 = qFinal2;
    q3 = qFinal3;
  }

  std::cout << "This is the final q" << std::endl;
  std::cout << "q0: " << q0 << std::endl;
  std::cout << "q1: " << q1 << std::endl;
  std::cout << "q2: " << q2 << std::endl;
  std::cout << "q3: " << q3 << std::endl;

  // save facs for each coordinate scaling
  const double fac_xx = q0*q0 + q1*q1 - q2*q2 - q3*q3;
  const double fac_xy = 2.0*(q1*q2 - q0*q3);
  const double fac_xz = 2.0*(q0*q2 + q1*q3);
  const double fac_yx = 2.0*(q1*q2 + q0*q3);
  const double fac_yy = q0*q0 - q1*q1 + q2*q2 - q3*q3;
  const double fac_yz = 2.0*(q2*q3 - q0*q1);
  const double fac_zx = 2.0*(q1*q3 - q0*q2);
  const double fac_zy = 2.0*(q0*q1 + q2*q3);
  const double fac_zz = q0*q0 - q1*q1 - q2*q2 + q3*q3;

  // test
  const double cX = 1.0;
  const double cY = 0.0;
  const double cZ = 0.0;
  
  // rotated model coordinates; converted to displacement; add back in centroid
  const double rcX = fac_xx*cX + fac_xy*cY + fac_xz*cZ;
  const double rcY = fac_yx*cX + fac_yy*cY + fac_yz*cZ;
  const double rcZ = fac_zx*cX + fac_zy*cY + fac_zz*cZ;
  std::cout << rcX << " " << rcY << " " << rcZ << std::endl;
  
  return 0;
  
}


