#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void
print_float(const char *ident, float x)
{
  union
  {
    float f;
    uint32_t i;
  } u;

  u.f = x;
  printf("%s = %016lx = %.17g\n", ident, u.i, x);
}

int
main(int argc, char **argv)
{
  float x = 55;
  float y = 0.69314718055994529;
  float z = 38.123094930796988;
  float dst;

  //32bit variant
  asm("fmsub %s0, %s1, %s2, %s3\n;" : "=w"(dst) : "w"(x), "w"(y), "w"(z));
  printf("FMSUB 32bit: dst = z + (-x) * y\n");
  printf("%f = %f + (-%f) * %f\n", dst, z, x, y);
  print_float("dst", dst);
  
  return 0;
}
