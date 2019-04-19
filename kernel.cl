#define DIM 5
#define STEP 1.e-4
#define RSTEP 1.e4
#define TOL 1.e-3
#define SQTOL 1.e-6
#define MAXSTEP 10
#define ASIZE 1.e3

typedef struct{
        unsigned int num;
        float start;
        float step;
        float end;
        __global float* a;
        __global float* b;
        __global float* c;
        __global float* d;
} disp_struct;

typedef struct{
        unsigned int num;
        float start;
        float step;
        float end;
} simple_disp_struct;

typedef struct{
        float e1;
        float e2;
        float h;
        float f;
        float p;
} vec;

typedef struct{
        disp_struct* ed;
        disp_struct* hd;
        disp_struct* fd;
} disp_pool;

__constant vec basis[] = {
        (vec) {1., 0., 0., 0., 0.},
        (vec) {0., 1., 0., 0., 0.},
        (vec) {0., 0., 1., 0., 0.},
        (vec) {0., 0., 0., 1., 0.},
        (vec) {0., 0., 0., 0., 1.}}; 

inline float absf(float a){
        return (a > 0.) ? a : -a;
}

inline float Dot(vec a, vec b){
        float* ap = (float*) &a;
        float* bp = (float*) &b;
        return (ap[0] * bp[0] 
                + ap[1] * bp[1]
                + ap[2] * bp[2] 
                + ap[3] * bp[3]
                + ap[4] * bp[4]);
}

inline vec VecSpread(float a){
        return (vec) {a, a, a, a, a};
}

inline vec Sum(vec a, vec b){
        float* ap = (float*) &a;
        float* bp = (float*) &b;
        return (vec){
                ap[0] + bp[0],
                ap[1] + bp[1],
                ap[2] + bp[2],
                ap[3] + bp[3],
                ap[4] + bp[4]};
}

inline vec Mul(float c, vec a){
        float* ap = (float*) &a;
        return (vec){
                ap[0] * c,
                ap[1] * c,
                ap[2] * c,
                ap[3] * c,
                ap[4] * c};
}

inline vec VecMul(vec a, vec b){
        float* ap = (float*) &a;
        float* bp = (float*) &b;
        return (vec){
                ap[0] * bp[0],
                ap[1] * bp[1],
                ap[2] * bp[2],
                ap[3] * bp[3],
                ap[4] * bp[4]};
}

inline vec Neg(vec a){
        return Mul(-1., a);
}

inline vec Sub(vec a, vec b){
        return Sum(a, Neg(b));
}

inline float Norm(vec a){
        return sqrt(Dot(a, a));
}

inline vec Normalize(vec a){
        return Mul(1. / Norm(a), a);
}

inline unsigned int GetNum(float arg, disp_struct* inp){
        float n = (arg - inp->start) / (inp->step);
        if(arg <= 0.) return 0;
        else if(arg >= inp->end) return inp->num - 1;
        else return (unsigned int) n;

}

inline float Spline(float arg,   disp_struct* inp){
        unsigned int num = GetNum(arg, inp);
	float x = arg - inp->step * (float) num;
        float res = ((inp->a[num] * x + inp->b[num]) * x + inp->c[num]) * x + inp->d[num];
        return res;
}

inline float DSpline(float arg,   disp_struct* inp){
        unsigned int num = GetNum(arg, inp);
	float x = arg - inp->step * (float) num;
        float res = (3. * inp->a[num] * x + 2. * inp->b[num]) * x + inp->c[num];
        return res;
}

inline float Function(vec point, disp_pool* dspl){
        float* pp = (float*) &point;
        float psum = pp[0] + pp[1] + pp[2];
        float a, b, c, d;
        a = Spline(pp[0], dspl->ed) + Spline(pp[1], dspl->ed)
                - Spline(pp[2], dspl->hd) - Spline(psum, dspl->fd);
        b = DSpline(psum, dspl->fd) - pp[3] * 
                (Spline(pp[0], dspl->ed) - Spline(psum, dspl->fd));
        c = DSpline(psum, dspl->ed) - pp[3] * 
                (Spline(pp[1], dspl->ed) - Spline(psum, dspl->fd));
        d = DSpline(psum, dspl->fd) + pp[3] * 
                (Spline(pp[2], dspl->hd) + Spline(psum, dspl->fd));
        return (a * a + b * b + c * c + d * d + pp[4] * pp[4]);
}


//4 point derrivate formula
inline float Diff1D(vec point, int i,   disp_pool* spl){
        float* pp = (float*) &point;
        float mstep = STEP * (absf(pp[i]) + 1.);
        float ret_val = Function(
                Sum(point, Mul(-2. * mstep, basis[i])), spl);
        ret_val -= 8. * Function(
                Sum(point, Mul(-mstep, basis[i])), spl);
        ret_val += 8. * Function(
                Sum(point, Mul(mstep, basis[i])), spl);
        ret_val -= Function(
                Sum(point, Mul(2. * mstep, basis[i])), spl);
        return (ret_val / (12. * mstep));
}


inline vec DiffND(vec point,   disp_pool* spl){
        return (vec){
                Diff1D(point, 0, spl),
                Diff1D(point, 1, spl),
                Diff1D(point, 2, spl),
                Diff1D(point, 3, spl),
                Diff1D(point, 4, spl)};
}

inline float DiffDir(vec point, vec direction,   disp_pool* spl){
        vec ndir = Normalize(direction);
        return Dot(ndir, DiffND(point, spl));
}

inline float DDiffDir(vec point, vec direction, disp_pool* spl){
        vec ndir = Normalize(direction);
        float mstep = STEP * (Norm(point) + 1.);
        float ret_val = Dot(ndir, DiffND(
                Sum(Mul(mstep, ndir), point), spl));
        ret_val -= Dot(ndir, DiffND(
                Sum(Mul(-mstep, ndir), point), spl));
        return (ret_val / mstep);
}

vec Descent1D(vec point, vec direction, disp_pool* spl){
        vec ret_val = point;
        vec ndir = Normalize(direction);
        float di, ddi;
        for(unsigned int i = 0; i < MAXSTEP; i++){
                di = DiffDir(ret_val, direction, spl);
                ddi = DDiffDir(ret_val, direction, spl);
                if(Function(ret_val, spl) < SQTOL) break;
                ret_val = Sum(ret_val, Mul((- di / ddi), ndir));
        }
        return ret_val;
}


vec NonlinearCG(vec point,   disp_pool* spl){
        float alpha, beta, pgradsqnorm, gradsqnorm;
        vec opt;
        vec loc = point;
        vec grad = DiffND(loc, spl);
        vec pgrad;
        vec s = Neg(grad);
        pgradsqnorm = Dot(grad, grad);
        unsigned int i;
        for(i = 0; i < MAXSTEP; i++){
                pgrad = grad;
                grad = DiffND(loc, spl);
                gradsqnorm = Dot(grad, grad);
                if(Function(loc, spl) < SQTOL) break;
                beta = Dot(grad, Sub(grad, pgrad)) / pgradsqnorm;
                pgradsqnorm = gradsqnorm;
                s = Sum(Neg(grad), Mul(beta, s));
                opt = Descent1D(loc, s, spl);
                alpha = Norm(Sub(opt, loc)) / Norm(s);
                loc = Sum(loc, Mul(alpha, s));
        }
        return loc;
}

disp_struct Decompile(__global simple_disp_struct* input){
        size_t array_size = input->num * sizeof(float);
        __global void* now = (__global void*) input + sizeof(simple_disp_struct);
        __global void* locs[4];
        for(int i = 0; i < 4; i++){
                locs[i] = now;
                now += array_size;
        }
        disp_struct result = {input->num, input->start, input->step, 
                input->end, locs[0], locs[1], locs[2], locs[3]};
        return result;
}

__kernel void the_main_void(__global simple_disp_struct* e_disp,
        __global simple_disp_struct* h_disp,
        __global simple_disp_struct* f_disp,
        __global vec* initp, __global vec* output,
        __global float* outcharval){
        disp_struct ed = Decompile(e_disp);
        disp_struct hd = Decompile(h_disp);
        disp_struct fd = Decompile(f_disp);
        disp_pool dp = (disp_pool){&ed, &hd, &fd};
        int gid = get_global_id(0);
        output[gid] = NonlinearCG(initp[gid], &dp);
        outcharval[gid] = Function(output[gid], &dp);
}