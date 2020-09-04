#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void
print_float(const char *ident, double x)
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
  double x = strtod(argv[1], NULL);
  double y = strtod(argv[2], NULL);
  double z = strtod(argv[3], NULL);
  double dst;
  float fx = x;
  float fy = y;
  float fz = z;
  float fdst;

  print_float("x", x);
  print_float("y", y);
  print_float("z", z);

  dst = __builtin_fma (x, y, -z);
  print_float("dst", dst);

  //dst = __builtin_fmaf (x, y, -z);
  //print_float("dst_fmaf", dst);

  asm("fmsub %d0, %d1, %d2, %d3\n;" : "=w"(dst) : "w"(x), "w"(y), "w"(z));
  printf("dst_fmsub %f, x %f\n", dst, x);
  //32bit variant
  asm("fmsub %s0, %s1, %s2, %s3\n;" : "=w"(fdst) : "w"(fx), "w"(fy), "w"(fz));
  printf("fdst_fmsub %f, x %f\n", fdst, fx);

  asm("fnmadd %d0, %d1, %d2, %d3\n;" : "=w"(dst) : "w"(x), "w"(y), "w"(z));
 // print_float("fnmadd", dst);
  printf("dst_fmnadd %f, x %f, y %f, z %f\n", dst, x, y, z);
  asm("fnmadd %s0, %s1, %s2, %s3\n;" : "=w"(fdst) : "w"(fx), "w"(fy), "w"(fz));
  printf("fdst_fmnadd %f, x %f\n", fdst, fx);
  asm("fnmsub %d0, %d1, %d2, %d3\n;" : "=w"(dst) : "w"(x), "w"(y), "w"(z));
  print_float("dst_fnmsub", dst);
  asm("fnmsub %s0, %s1, %s2, %s3\n;" : "=w"(fdst) : "w"(fx), "w"(fy), "w"(fz));
  printf("fdst_fmnsub %f, x %f\n", fdst, fx);
  
  return 0;
}
