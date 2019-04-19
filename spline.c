#include <stdlib.h> 
#include <stdio.h>
#include "spline.h"

/*
https://gist.github.com/svdamani/1015c5c4b673c3297309
*/



disp_struct UnNormSpline(unsigned int n, float* xs, float* ys){
	n--;
	int i, j;
	disp_struct ret_val;
	float* 	x = (float*) malloc((n + 1) * sizeof(float));
	float* 	a = (float*) malloc((n + 1) * sizeof(float));
	float* 	h = (float*) malloc(n * sizeof(float));
	float* 	A = (float*) malloc(n * sizeof(float));
	float* 	l = (float*) malloc((n + 1) * sizeof(float));
	float* 	u = (float*) malloc((n + 1) * sizeof(float));
	float* 	z = (float*) malloc((n + 1) * sizeof(float));
	float* 	c = (float*) malloc((n + 1) * sizeof(float));
	float* 	b = (float*) malloc(n * sizeof(float));
	float* 	d = (float*) malloc(n * sizeof(float));
	for(i = 0; i < (n + 1); i++){
		x[i] = xs[i];
		a[i] = ys[i];
	}
	/** Step 1 */
    for (i = 0; i <= n - 1; ++i) h[i] = x[i + 1] - x[i];
    /** Step 2 */
    for (i = 1; i <= n - 1; ++i)
        A[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1];
    /** Step 3 */
    l[0] = 1;
    u[0] = 0;
    z[0] = 0;
    /** Step 4 */
    for (i = 1; i <= n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i]) - h[i - 1] * u[i - 1];
        u[i] = h[i] / l[i];
        z[i] = (A[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    /** Step 5 */
    l[n] = 1;
    z[n] = 0;
    c[n] = 0;
    /** Step 6 */
    for (j = n - 1; j >= 0; --j) {
        c[j] = z[j] - u[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }
	ret_val = (disp_struct){n, x[0], (x[n] - x[0]) / n, x[n], d, c, b, a};
	free(x); free(h); free(A); free(l); free(u); free(z);
	return ret_val;
}



/*int main(){
	int n, i, j;
	scanf("%d", &n);
	n--;
	float x[n + 1] = {0., 1., 2.}
	disp_struct d = UnNormSpline(n, x, a);
	printf("%2s %8s %8s %8s %8s\n", "i", "ai", "bi", "ci", "di");
    for (i = 0; i < n; ++i)
        printf("%2d %8.2f %8.2f %8.2f %8.2f\n", i, d.a[i], d.b[i], d.c[i], d.d[i]);
	return n;
}*/

unsigned int GetNum(float arg, disp_struct* inp){
        float n = (arg - inp->start) / (inp->step);
        if(arg <= 0.) return 0;
        else if(arg >= inp->end) return inp->num - 1;
        else return (unsigned int) n;

}
                
float Spline(float arg, disp_struct* inp){
        unsigned int num = GetNum(arg, inp);
		float x = arg - inp->step * (float) num;
        float res = ((inp->a[num] * x + inp->b[num]) * x + inp->c[num]) * x + inp->d[num];
        return res;
}