all:
	clang -g main.c compile_data.c spline.c -o oger -std=c99 -D DEBUG -framework OpenCL -v