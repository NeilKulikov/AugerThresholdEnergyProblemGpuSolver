#include "copcl.h"
#include <string.h>
#include <stdlib.h>
#ifdef DEBUG
    #include <stdio.h>
#endif


mem_range compile_disp(disp_struct arg){
    size_t array_size = arg.num * sizeof(float);
    size_t data_size = sizeof(simple_disp_struct) + 4 * array_size;
    void* location = malloc(data_size + 64);
    void* now = location;
    mem_range result = {data_size, location};
    void* locs[] = {arg.a, arg.b, arg.c, arg.d};
    memcpy(now, &arg, sizeof(simple_disp_struct));
    now += sizeof(simple_disp_struct);
    for(u64 i = 0; i < 4; i++){
        if(locs[i] == NULL){
            #ifdef DEBUG
                printf("Warning: Compiled unfully!\n");
            #endif
            return result;
        }
        memcpy(now, locs[i], array_size);
        now += array_size;
    }
    #ifdef DEBUG
        printf("Success: Compiled fully! Location: %zu \n", result.host_location);
    #endif
    return result;
}

disp_struct Decompile(simple_disp_struct* input){
        size_t array_size = input->num * sizeof(float);
        void* now = (void*) input + sizeof(simple_disp_struct);
        void* locs[4];
        for(int i = 0; i < 4; i++){
                locs[i] = now;
                now += array_size;
        }
        disp_struct result = {input->num, input->start, input->step, 
                input->end, locs[0], locs[1], locs[2], locs[3]};
        return result;
}