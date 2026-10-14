 gcc -c -O2 gsw_oceanographic_toolbox.c
 gcc -c -O2 gsw_saar.c
 g++ -O2 cce1_teos10_density.cpp     gsw_oceanographic_toolbox.o     gsw_saar.o     -o cce1_teos10_density

