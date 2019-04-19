#ifndef COPCL
#define COPCL
#include <stddef.h>
#include "defs.h"

typedef struct{
        unsigned int num;
        float start;
        float step;
        float end;
        float* a;
        float* b;
        float* c;
        float* d;
} disp_struct;

typedef struct{
        unsigned int num;
        float start;
        float step;
        float end;
} simple_disp_struct;

typedef struct{
    size_t length;
    void* host_location;
} mem_range;

typedef struct{
        float e1;
        float e2;
        float h;
        float f;
        float p;
} vec;

extern mem_range memalloc(size_t);
extern mem_range compile_disp(disp_struct);
extern disp_struct Decompile(simple_disp_struct*);
#endif