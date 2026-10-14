#include <iostream>
#include <cmath>
#include <iomanip>


struct Vec3
{
    double x;
    double y;
    double z;
};


Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return {
        a.x+b.x,
        a.y+b.y,
        a.z+b.z
    };
}

Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return {
        a.x-b.x,
        a.y-b.y,
        a.z-b.z
    };
}

Vec3 operator*(double s, const Vec3& a)
{
    return {
        s*a.x,
        s*a.y,
        s*a.z
    };
}


double dot(const Vec3& a, const Vec3& b)
{
    return
        a.x*b.x +
        a.y*b.y +
        a.z*b.z;
}


double magnitude(const Vec3& a)
{
    return std::sqrt(dot(a,a));
}


Vec3 normalize(const Vec3& a)
{
    double mag = magnitude(a);

    return
    {
        a.x/mag,
        a.y/mag,
        a.z/mag
    };
}

void printVector(const char* name,
                 const Vec3& a)
{
    std::cout
        << name
        << " = ( "
        << a.x << ", "
        << a.y << ", "
        << a.z
        << " )\n";
}


int main()
{
  std::cout
    << std::setprecision(6);

  // inlet velocity magnitude
  double UwMag = 0.20; 
  
  // fish velocity magnitude
  double M = 0.50;

  // water velocity
  Vec3 nW =
    {
      0.0,
      1.0,
      0.0
    };
  nW = normalize(nW);
  Vec3 Uw = UwMag*nW;
  
  // e_in : supplied CFD inlet direction
  Vec3 eIn =
    {
      0.0,
      1.0,
      0.0
    };
  eIn = normalize(eIn);
  
  // n : fish swimming direction
  const double sqrtTwo = std::sqrt(2.0);
  Vec3 nFish =
    {
      0.0,
      -1.0/sqrtTwo,
      -1.0/sqrtTwo
    };
  nFish = normalize(nFish);
  
  // Fish velocity 
  Vec3 Vf = M*nFish;
  
  // Apparent velocity
  Vec3 Uapp = Uw - Vf;

  // Normalized apparent velocity
  Vec3 uAppHat =
    normalize(Uapp);
  
  // Apparent velocity magnitude
  double Uinf =
    magnitude(Uapp);

  // CFD inlet velocity
  Vec3 UinfCFD = Uinf*eIn;
    
  // OUTPUT
  printVector("Uw",Uw);
  
  printVector("Vf",Vf);

  printVector("nFish",nFish);
  
  printVector("Uapp",Uapp);

  Uapp = normalize(Uapp);

  printVector("nApp", Uapp);
 
  std::cout
    << "Uinf = "
    << Uinf
    << "\n\n";
    
  printVector("eIn",eIn);
  
  printVector("UinfCFD",UinfCFD);
  
  return 0;
}
