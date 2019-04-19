#include "copcl.h"
#ifndef SPLINE
#define SPLINE
extern disp_struct UnNormSpline(unsigned int, float*, float*);
extern disp_struct NormalizeSpline(unsigned int, float*, float*);
extern float Spline(float, disp_struct*);
#endif