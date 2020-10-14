#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void
print_double(const char *ident, double x)
{
  union
  {
    double f;
    uint64_t i;
  } u;

  u.f = x;
  printf("%s = %016lx = %.17g\n", ident, u.i, x);
}

int
main(int argc, char **argv)
{
  double x = 55;
  double y = 0.69314718055994529;
  double z = 38.123094930796988;
  double dst;

  asm("fmsub %d0, %d1, %d2, %d3\n;" : "=w"(dst) : "w"(x), "w"(y), "w"(z));
  printf("FMSUB 64bit: dst = z + (-x) * y\n");
  printf("%f = %f + (-%f) * %f\n", dst, z, x, y);
  print_double("dst", dst);
  
  return 0;
}
