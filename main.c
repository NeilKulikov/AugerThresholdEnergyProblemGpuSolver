#include <stdio.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include "copcl.h"
#include "spline.h"

typedef struct{
    char* data;
    size_t length;
} string;

typedef struct{
    disp_struct ed;
    disp_struct hd;
    disp_struct fd;
} disp_pool;

const char* file_name = "kernel.cl";
int err = 0;

string read_whole_file(const char* fname){
    FILE* file_obj = fopen(fname, "rb");
    fseek(file_obj, 0 , SEEK_END);
    u64 file_size = ftell(file_obj);
    rewind(file_obj);
    char* pre_result = malloc((size_t) file_size + 1);
    pre_result[file_size] = 0;
    fread(pre_result, file_size, 1, file_obj); 
    fclose(file_obj);
    string result = {pre_result, file_size + 1};
    return result;
}

cl_device_id get_device(int is_gpu){
    cl_device_id result;
    err = clGetDeviceIDs(NULL, is_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &result, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
    }
    return result;
}

cl_context create_context(cl_device_id device_id){
    cl_context result = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!result){
        printf("Error: Failed to create a compute context!\n");
    }
    return result;
}

cl_program create_program(cl_context context, string kernel_source){
    cl_program result = clCreateProgramWithSource(context, 1, (const char**) &(kernel_source.data), (const size_t*) &(kernel_source.length), &err);
    if (!result){
        printf("Error: Failed to create compute program!\n");
    }
    return result;
}

cl_command_queue create_cqueue(cl_context context, cl_device_id device_id){
    cl_command_queue result = clCreateCommandQueue(context, device_id, 0, &err);
    if(!result){
        printf("Error: Failed to create a command commands!\n");
    }
    return result;
}

void build_program(cl_program program, cl_device_id device_id){
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[10000];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }
}

cl_kernel create_kernel(cl_program program){
    cl_kernel result = clCreateKernel(program, "the_main_void", &err);
    if (!result || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
    }
    return result;
}

cl_mem create_buffer(cl_context context, int type, size_t size){
    cl_mem result = clCreateBuffer(context, type, size, NULL, NULL);
    if(!result){
        printf("Error: Failed to create bufferl!\n");
    }
    #ifdef DEBUG
        printf("Success: buffer created!\n");
    #endif
    return result;
}

void write_buffer(cl_command_queue commands, cl_mem buffer, void* data, size_t size){
    err = clEnqueueWriteBuffer(commands, buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to write to source array!\n");
    }
}

/*void* read_buffer(cl_command_queue commands, cl_mem buffer, size_t size){
    void* result = memalloc(size).host_location;
    err = clEnqueueReadBuffer(commands, buffer, CL_TRUE, 0, size, result, 0, NULL, NULL);  
    if (err != CL_SUCCESS){
        printf("Error: Failed to read output array! %d\n", err);
    }
    return result;
}*/

void* read_buffer(cl_command_queue commands, cl_mem buffer, mem_range result){
    err = clEnqueueReadBuffer(commands, buffer, CL_TRUE, 0, result.length, result.host_location, 0, NULL, NULL);  
    if (err != CL_SUCCESS){
        printf("Error: Failed to read output array! %d\n", err);
    }
    return result.host_location;
}

void set_arg(cl_kernel kernel, int num, cl_mem data){
    err  = clSetKernelArg(kernel, num, sizeof(cl_mem), &data);
    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments!\n");
    }
}

size_t get_num_pu(cl_kernel kernel, cl_device_id device_id){
    size_t result;
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(result), &result, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    }
    return result;
}

void execute_kernel(cl_kernel kernel, cl_command_queue commands, size_t task_size, size_t pu_size){
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &task_size, &pu_size, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to execute kernel!\n");
    }
}

void finish_task(cl_command_queue commands){
    err = clFinish(commands);
    if (err != CL_SUCCESS){
        printf("Error: Failed to execute kernel!\n");
    }
}

mem_range memalloc(size_t size){
    void* ret_val = malloc(size + 16);
    if(ret_val == NULL)
        printf("Error: Failed to allocate memory!\n");
    #ifdef DEBUG
        printf("Success: memory alocated!!!!\n");
    #endif
    return (mem_range) {size, ret_val};
}

float gen_rand(){
    return ((float) rand()) / ((float) RAND_MAX);
}

float gen_norm_rand(){
    float a = gen_rand(), b = gen_rand();
    float s = a * a + b * b;
    return a * sqrtf((-2. * logf(s))/ s);
}

vec gen_rand_vec(){
    return (vec) {
        gen_rand() * 10. - 5.,
        gen_rand() * 10. - 5.,
        gen_rand() * 10. - 5.,
        gen_rand() * 10.,
        gen_rand() * 10.};
}

disp_pool parse_file(FILE* file){
        if(file == NULL) printf("Error");
        #ifdef DEBUG
            printf("Disp file opened\n");
        #endif
        int num = 0;
        float k, e, h, f;
        float* ks = (float*) malloc(128 * sizeof(float));
        float* es = (float*) malloc(128 * sizeof(float));
        float* hs = (float*) malloc(128 * sizeof(float));
        float* fs = (float*) malloc(128 * sizeof(float));
        while(fscanf(file, "%f %f %f %f", &(ks[num]),
            &(es[num]), &(hs[num]), &(fs[num])) != EOF){
            printf("%f %f %f %f\n", ks[num], es[num], hs[num], fs[num]);
            num++;
            if((num != 0) && !(num % 128)){
                ks = realloc((void*) ks, (num + 128) * sizeof(float));
                es = realloc((void*) es, (num + 128) * sizeof(float));
                hs = realloc((void*) hs, (num + 128) * sizeof(float));
                fs = realloc((void*) fs, (num + 128) * sizeof(float));
            }
        }
        disp_pool ret_val;
        ret_val.ed = UnNormSpline(num, ks, es);
        ret_val.hd = UnNormSpline(num, ks, hs);
        ret_val.fd = UnNormSpline(num, ks, fs);
        return ret_val;
}

int main(int argc, char* argv[]){
    int msteps = 5;
    char* dfname = "./test_data.txt";
    if(argc >= 1) msteps = atoi(argv[1]);
    printf("Max steps: %i\n", msteps);
    if(argc >= 2) dfname = argv[2];
    printf("File name: %s\n", dfname);
    FILE* dfile = fopen(dfname, "rb");
    disp_pool dp = parse_file(dfile);
    fclose(dfile);
    srand(time(NULL));
    string kernel_source = read_whole_file(file_name);
    #ifdef DEBUG
        printf("Success: Kernel file have been read!\n");
    #endif
    cl_device_id device = get_device(1);
    #ifdef DEBUG
        printf("Success: GPU found!\n");
    #endif
    cl_context context = create_context(device);
    #ifdef DEBUG
        printf("Success: Context created!\n");
    #endif
    cl_program program = create_program(context, kernel_source);
    #ifdef DEBUG
        printf("Success: Program created!\n");
    #endif
    cl_command_queue commands = create_cqueue(context, device);
    #ifdef DEBUG
        printf("Success: Command queue created!\n");
    #endif
    build_program(program, device);
    #ifdef DEBUG
        printf("Success: Program compiled!\n");
    #endif
    cl_kernel kernel = create_kernel(program);
    #ifdef DEBUG
        printf("Success: Kernel created!\n");
    #endif
    size_t local_pu_num = get_num_pu(kernel, device);
    #ifdef DEBUG
            printf("Success: Found %zu processor units!\n", local_pu_num);
    #endif
    mem_range ed = compile_disp(dp.ed);
    mem_range hd = compile_disp(dp.hd);
    mem_range fd = compile_disp(dp.fd);
    #ifdef DEBUG
        printf("Success: disp compiled!\n");
    #endif
    cl_mem ed_in = create_buffer(context, CL_MEM_READ_ONLY, ed.length + 256);
    cl_mem hd_in = create_buffer(context, CL_MEM_READ_ONLY, hd.length + 256);
    cl_mem fd_in = create_buffer(context, CL_MEM_READ_ONLY, fd.length + 256);
    cl_mem initp_in = create_buffer(context, CL_MEM_READ_WRITE, local_pu_num * sizeof(vec) + 256);
    cl_mem simp_out = create_buffer(context, CL_MEM_READ_WRITE, local_pu_num * sizeof(vec) + 256);
    cl_mem char_out = create_buffer(context, CL_MEM_READ_WRITE, local_pu_num * sizeof(float) + 256);
    //cl_mem ener_out = create_buffer(context, CL_MEM_READ_WRITE, local_pu_num * sizeof(float) + 256);
    #ifdef DEBUG
        printf("Success: Memory allocated!\n");
    #endif
    write_buffer(commands, ed_in, ed.host_location, ed.length);
    write_buffer(commands, hd_in, hd.host_location, hd.length);
    write_buffer(commands, fd_in, fd.host_location, fd.length);
    #ifdef DEBUG
        printf("Success: disps written!\n");
    #endif
    #ifdef DEBUG
        clock_t timer = clock();
    #endif
    vec best_point = (vec) {0., 0., 0., 0., 0.};
    float best_energy = 1.e8, best_char = 1.e8;
    mem_range outp = memalloc(local_pu_num * sizeof(vec));
    mem_range initp = memalloc(local_pu_num * sizeof(vec));
    mem_range charval = memalloc(local_pu_num * sizeof(float));
    //mem_range enerval = memalloc(local_pu_num * sizeof(float));
    for(int i = 0; i < local_pu_num; i++) ((vec*) initp.host_location)[i] = gen_rand_vec();
    //float 
    for(int i = 0; i < msteps; i++){
        #ifdef DEBUG
            printf("Iter: %i\n", i);
        #endif
        write_buffer(commands, initp_in, initp.host_location, initp.length);
        #ifdef DEBUG
            printf("Success: Memory copied!\n");
        #endif
        set_arg(kernel, 0, ed_in);
        set_arg(kernel, 1, hd_in);
        set_arg(kernel, 2, fd_in);
        set_arg(kernel, 3, initp_in);
        set_arg(kernel, 4, simp_out);
        set_arg(kernel, 5, char_out);
        //set_arg(kernel, 6, ener_out);
        #ifdef DEBUG
            printf("Success: Arguments seted!\n");
        #endif
        execute_kernel(kernel, commands, local_pu_num, local_pu_num);
        #ifdef DEBUG
            printf("Success: Kernel execution started!\n");
        #endif
        finish_task(commands);
        #ifdef DEBUG
            printf("Success: Kernel execution ended!\n");
        #endif
        read_buffer(commands, simp_out, outp);
        read_buffer(commands, char_out, charval);
        //read_buffer(commands, ener_out, enerval);
        #ifdef DEBUG
            printf("Success: Buffer readed!\n");
        #endif
        float* outg = (float*) charval.host_location;
        vec* outv = (vec*) outp.host_location;
        float psum, energy;
        for(int j = 0; j < local_pu_num; j++){
            psum = outv[j].e1 + outv[j].e2 + outv[j].h;
            energy = Spline(psum, (disp_struct*) &(dp.fd));
            if(energy < best_energy){
                best_energy = energy;
                best_char = outg[j];
                best_point = outv[j];
            }
        }
        printf("\n\tBest energy: %e\n", energy);
        printf("\n\tIt's char: %e\n", best_char);
        printf("\tIt's vec: [0]:%e\t[1]:%e\t[2]:%e\t[3]:%e\t[4]:%e\n\n",
            best_point.e1, best_point.e2, best_point.h, best_point.f, best_point.p);
        initp = outp;
    }
    #ifdef DEBUG
        timer = clock() - timer;
        unsigned int msec = timer * 1000 / CLOCKS_PER_SEC;
        printf("Computation taken %u s  %u ms!\n", msec / 1000, msec % 1000);
    #endif
    free(initp.host_location);
    free(ed.host_location);
    free(hd.host_location);
    free(fd.host_location);
    //
    clReleaseMemObject(ed_in);
    clReleaseMemObject(hd_in);
    clReleaseMemObject(fd_in);
    clReleaseMemObject(initp_in);
    clReleaseMemObject(simp_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    #ifdef DEBUG
        printf("Success: Application closed!\n");
    #endif
    return 0;
}