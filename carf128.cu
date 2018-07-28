//
// Created by Yorrick on 28.04.2018.
//
// Functions for CARF with uint128_t.
//

#include <random>
#include "hpc_helpers.hpp" //https://github.com/JGU-HPC/parallelprogrammingbook/blob/master/include/hpc_helpers.hpp
#include "uint128_t.cu"

/**
 * Best settings for Tesla:
 * (no multiGPU / no streams)
 * NUMINPUTLINES 198647808 for max memory usage on Tesla
 * NUMTHREADS 128 for first kernel (89ms)
 * NUMTHREADS 608 for second kernel (1772.65ms) // 608 1772.65ms // 512 1798.48ms //256 1842.08ms // 768 1900ms //
 *
 * Best settings for Volta:
 * NUMINPUTLINES (190447616)
 * NUMTHREADS CARF: (352, 480, ) 768
 * NUMTHREADS popCount: 1024 (32 = 125ms to 1024 = 121)
 */
#define NUMINPUTLINES 198647808
#define NUMTHREADS (1024)

#define NUMTHREADS1 (128)
#define NUMTHREADS2 (608)

#define REPEATS (19000)
#define NUMBLOCKS 10


///////////////////////////////////////////////////////////////////////////////
//CARF - KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * Amends a single Hamming Mask, removing spurious 0s.
 */
__device__ uint128_t SHMS(uint128_t RH, uint128_t RL, uint128_t GH, uint128_t GL){
    return ((((RH ^ GH) | (RL ^ GL))<<1 & ((RH ^ GH) | (RL ^ GL))>>1) | (((RH ^ GH) | (RL ^ GL))<<1 & ((RH ^ GH) | (RL ^ GL))>>2) | (((RH ^ GH) | (RL ^ GL))<<2 & ((RH ^ GH) | (RL ^ GL))>>1) | ((RH ^ GH) | (RL ^ GL)));
}

/**
 * Computes the final bit-vector / Hamming Mask for the given pair using SHMS.
 * Stores final bit-vector in HM_OUT.
 */
__global__ void CARF(uint128_t * RH, uint128_t * RL, uint128_t * GH, uint128_t * GL, uint128_t * HM_OUT){

    const size_t thid = blockDim.x*blockIdx.x + threadIdx.x;

    if(thid < NUMINPUTLINES){
        HM_OUT[thid] = SHMS(RH[thid], RL[thid], GH[thid], GL[thid]) &
                       SHMS((RH[thid]>>1), (RL[thid]>>1), GH[thid], GL[thid]) &
                       SHMS((RH[thid]>>2), (RL[thid]>>2), GH[thid], GL[thid]) &
                       SHMS((RH[thid]<<1), (RL[thid]<<1), GH[thid], GL[thid]) &
                       SHMS((RH[thid]<<2), (RL[thid]<<2), GH[thid], GL[thid]);
        //printf("%u\n", HM_OUT[thid]);
    }
}

/**
 * Computes the final bit-vector / Hamming Mask for the given pair using SHMS.
 * Stores final bit-vector in HM_OUT.
 * Allows a single thread to process REPEATS many pairs.
 */
__global__ void CARF_loop(uint128_t * RH, uint128_t * RL, uint128_t * GH, uint128_t * GL, uint128_t * HM_OUT){

    const uint64_t thid = (blockDim.x*blockIdx.x + threadIdx.x)*REPEATS;

    if(thid < NUMINPUTLINES){

        for(uint16_t i = 0; i<REPEATS; i++){
            HM_OUT[(thid+i)] = SHMS(RH[(thid+i)], RL[(thid+i)], GH[(thid+i)], GL[(thid+i)]) &
                               SHMS((RH[(thid+i)]>>1), (RL[(thid+i)]>>1), GH[(thid+i)], GL[(thid+i)]) &
                               SHMS((RH[(thid+i)]>>2), (RL[(thid+i)]>>2), GH[(thid+i)], GL[(thid+i)]) &
                               SHMS((RH[(thid+i)]<<1), (RL[(thid+i)]<<1), GH[(thid+i)], GL[(thid+i)]) &
                               SHMS((RH[(thid+i)]<<2), (RL[(thid+i)]<<2), GH[(thid+i)], GL[(thid+i)]);
        }
    }
}






///////////////////////////////////////////////////////////////////////////////
// Conservative Population Count - KERNEL
///////////////////////////////////////////////////////////////////////////////

/**
 * Implementation of the conservative population count for uint128_t.
 * The combination of IF-ELSE-conditions and computation which provided best speed.
 * Use this!
 * Output is stored in errorCount.
 */
__global__ void ConservativePopCount(uint128_t * hm, uint128_t * errorCount){

    const auto thid = blockDim.x * blockIdx.x + threadIdx.x;

    if(thid<NUMINPUTLINES){

        uint8_t streak = 0;
        uint8_t totalErrors = 0;

        for(uint8_t i = 0; i < 128; i++){

            streak = streak + (((hm[thid]>>i) & 1) != 0);
            if (i > 0 && streak > 0 and (((hm[thid]>>i) & 1) == 0) and ((hm[thid]>>(i-1) & 1) != 0)) {
                totalErrors += 1 +((streak + 1) / 3);
                streak = 0;
            } else if(i==127 and streak > 0 and (((hm[thid]>>i) & 1) != 0) and ((hm[thid]>>(i-1) & 1) != 0)){
                totalErrors += 1 +((streak + 1) / 3);
            }
        }
        errorCount[thid] = totalErrors;
    }
}


/**
 * Allows a single thread to process REPEATS-many pairs.
 */
__global__ void naive_ConservativePopCount_loop(uint128_t * hm, uint128_t * errorCount){

    const uint64_t thid = (blockDim.x*blockIdx.x + threadIdx.x)*REPEATS;

    if(thid<NUMINPUTLINES) {

        for (uint16_t j = 0; j < REPEATS; j++) {

            uint8_t streak = 0;
            uint8_t totalErrors = 0;

            for (uint8_t i = 0; i < 128; i++) {

                streak = streak + (((hm[thid] >> i) & 1) != 0);
                if (i > 0 && streak > 0 and (((hm[thid] >> i) & 1) == 0) and ((hm[thid] >> (i - 1) & 1) != 0)) {
                    totalErrors += 1 + ((streak + 1) / 3);
                    streak = 0;
                } else if (i == 127 and streak > 0 and (((hm[thid] >> i) & 1) != 0) and
                           ((hm[thid] >> (i - 1) & 1) != 0)) {
                    totalErrors += 1 + ((streak + 1) / 3);
                }
            }
            errorCount[thid] = totalErrors;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// END OF KERNELS
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Single threaded / host functions
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// Host version of CARF
///////////////////////////////////////////////////////////////////////////////

/**
 * Serial SHMS
 */
uint128_t serialSHMS(uint128_t RH, uint128_t RL, uint128_t GH, uint128_t GL) {
    return ((((RH ^ GH) | (RL ^ GL)) << 1 & ((RH ^ GH) | (RL ^ GL)) >> 1) |
            (((RH ^ GH) | (RL ^ GL)) << 1 & ((RH ^ GH) | (RL ^ GL)) >> 2) |
            (((RH ^ GH) | (RL ^ GL)) << 2 & ((RH ^ GH) | (RL ^ GL)) >> 1) | ((RH ^ GH) | (RL ^ GL)));
}

/**
 * Serial CARF
 */
void serial_CARF(uint128_t * RH, uint128_t * RL, uint128_t * GH, uint128_t * GL, uint128_t * serial_HM_OUT){


    for(size_t i = 0; i < NUMINPUTLINES; i++){

        serial_HM_OUT[i] = serialSHMS(RH[i], RL[i], GH[i], GL[i]) &
                           serialSHMS((RH[i]>>1), (RL[i]>>1), GH[i], GL[i]) &
                           serialSHMS((RH[i]>>2), (RL[i]>>2), GH[i], GL[i]) &
                           serialSHMS((RH[i]<<1), (RL[i]<<1), GH[i], GL[i]) &
                           serialSHMS((RH[i]<<2), (RL[i]<<2), GH[i], GL[i]);

    }
}

/**
 * Serial naive conservative population count.
 */
void serial_conservative_popcount(uint128_t * final_hm, uint128_t * errorCount) {
    for (size_t index = 0; index < NUMINPUTLINES; index++) {

        std::bitset<64> hm1 = final_hm[index].LEFT;
        std::bitset<64> hm2 = final_hm[index].RIGHT;
        std::bitset<128> hm;

        for(int i = 127; i>63; i--){
            hm.set(i, hm1[i]);
        }

        for(int i = 63; i >= 0; i--){
            hm.set(i, hm2[i]);
        }

        size_t totalErrors = 0;
        size_t streak = 0;

        for (uint8_t i = 0; i < 128; i++) {
            if (hm[i] == 1) {
                streak++;
            }
            if (i > 0 and hm[i] == 0 and hm[i - 1] == 1) {
                totalErrors += 1 + ((streak + 1) / 3);
                streak = 0;
            } else if (i == 127 and streak > 0 and ((hm[i]) == 1) and ((hm[i - 1] == 1))) {
                totalErrors += 1 + ((streak + 1) / 3);
            }
        }
        errorCount[index] = totalErrors;
    }
}

///////////////////////////////////////////////////////////////////////////////
// I/O-functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Generates NUMININPUTLINES many completely random pairs in the given arrays.
 */
void generate_random_input(uint128_t * rh, uint128_t * rl, uint128_t * gh, uint128_t * gl){

    std::mt19937_64 gen (std::random_device{}());
    std::uint64_t randomNumber = gen();

    for(size_t i = 0; i < NUMINPUTLINES; i++){
        uint128_t a(gen(), gen());
        uint128_t b(gen(), gen());
        uint128_t c(gen(), gen());
        uint128_t d(gen(), gen());
        rh[i] = a;
        rl[i] = b;
        gh[i] = c;
        gl[i] = d;
    }

//    cout << "\nPrinting inputs:" << endl;
//    for (int i = 0; i < 7; i++){
//        cout << "rh["<< i <<"] "; rh[i].printBits();
//        cout << "rl["<< i <<"] "; rl[i].printBits();
//        cout << "gh["<< i <<"] "; gh[i].printBits();
//        cout << "gl["<< i <<"] "; gl[i].printBits();
//    }
//    cout << endl;

}

size_t serial_countExceedingThreshold(uint128_t * errorCount, uint8_t threshold){
    size_t negatives = 0;

    for(int i = 0; i < NUMINPUTLINES; i++){
        //errorCount[i].print();cout<<endl;
        if(errorCount[i]>threshold){
            negatives++;
        }
    }

    cout << "Negatives: " << negatives << endl;
    return negatives;
}

///////////////////////////////////////////////////////////////////////////////
// Debug functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Compares the final bit-vectors computed by carf and serial carf.
 * Prints differing vectors to console.
 */
void compare_results(uint128_t * serial_hm_out, uint128_t * parallel_hm_out){
    size_t errorcount = 0;
    for(size_t i = 0; i < NUMINPUTLINES; i++){

        if(serial_hm_out[i]!=parallel_hm_out[i]){
            errorcount++;
            cout << "Error in entry " << i << ": serial: "; serial_hm_out[i].print(); cout << " parallel: "; parallel_hm_out[i].print(); cout <<endl;
        }
    }
    cout << "Found errors: " << errorcount << endl;
}

/**
 * Prints device info.
 */
void printDeviceInfo(){
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}


///////////////////////////////////////////////////////////////////////////////
// Sequences
///////////////////////////////////////////////////////////////////////////////


/**
 * Test-Code for uint128_t
 */
void kernel128_test() {

    cout << "Starting kernel128 test." << endl;

    TIMERSTART(total_kernel_test);
    uint128_t *rh = new uint128_t[NUMINPUTLINES];
    uint128_t *rl = new uint128_t[NUMINPUTLINES];
    uint128_t *gh = new uint128_t[NUMINPUTLINES];
    uint128_t *gl = new uint128_t[NUMINPUTLINES];
    uint128_t *serial_hm_out = new uint128_t[NUMINPUTLINES];
    uint128_t *parallel_hm_out = new uint128_t[NUMINPUTLINES];

    uint128_t *RH = nullptr, *RL = nullptr, *GH = nullptr, *GL = nullptr;

    TIMERSTART(generate_random_input);
    generate_random_input(rh, rl, gh, gl);
    TIMERSTOP(generate_random_input);

    rh[0] = rl[0] = gh[0] = gl[0] = 0;

    TIMERSTART(cudaMalloc);
    cudaMalloc(&RH, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&RL, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&GH, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&GL, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;

    TIMERSTOP(cudaMalloc);

    TIMERSTART(cudaMemcpy);
    cudaMemcpy(RH, rh, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(RL, rl, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(GH, gh, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(GL, gl, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    TIMERSTOP(cudaMemcpy);


    TIMERSTART(kernel_CARF128);
    CARF<<<SDIV(NUMINPUTLINES, NUMTHREADS1), NUMTHREADS1>>>(RH, RL, GH, GL, RL);  CUERR;
    //CARF_loop<<<NUMBLOCKS, NUMTHREADS>>>(RH, RL, GH, GL, HM_OUT); CUERR;
    TIMERSTOP(kernel_CARF128);

    TIMERSTART(cudaMemcpy2);
    cudaMemcpy(parallel_hm_out, RL, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpy2);

    TIMERSTART(serial_algorithm);
    serial_CARF(rh, rl, gh, gl, serial_hm_out);
    TIMERSTOP(serial_algorithm);

    TIMERSTART(comparison);
    compare_results(serial_hm_out, parallel_hm_out);
    TIMERSTOP(comparison);
    serial_hm_out[(NUMINPUTLINES-1)].print();
    cout << "=" << endl;
    parallel_hm_out[(NUMINPUTLINES-1)].print();
    cout << endl;

    uint128_t * serial_errorCount = new uint128_t[NUMINPUTLINES];
    uint128_t * parallel_errorCount = new uint128_t[NUMINPUTLINES];

    TIMERSTART(cudaMemcpy3);
    cudaMemcpy(RH, parallel_errorCount, NUMINPUTLINES*sizeof(uint128_t), cudaMemcpyHostToDevice); CUERR;
    TIMERSTOP(cudaMemcpy3);

    TIMERSTART(parallel_popCount);
    ConservativePopCount<<<SDIV(NUMINPUTLINES, NUMTHREADS2), NUMTHREADS2>>>(RL, RH); CUERR;
    //ConservativePopCount_loop<<<NUMBLOCKS, NUMTHREADS>>>(RL, RH); CUERR;
    TIMERSTOP(parallel_popCount);

    TIMERSTART(cudaMemcpy4);
    cudaMemcpy(parallel_errorCount, RH, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpy4);

    TIMERSTART(serial_conservativePopC);
    serial_conservative_popcount(serial_hm_out, serial_errorCount);
    TIMERSTOP(serial_conservativePopC);



    TIMERSTART(comparison_errorrate);
    size_t s = serial_countExceedingThreshold(serial_errorCount, 2);
    size_t p = serial_countExceedingThreshold(parallel_errorCount, 2);

    cout << s << "=" << p << endl;
    serial_errorCount[(NUMINPUTLINES-1)].print();
    cout << "=";
    parallel_errorCount[(NUMINPUTLINES-1)].print();
    cout << endl;

    TIMERSTOP(comparison_errorrate);

    cudaFree(RH);
    cudaFree(RL);
    cudaFree(GH);
    cudaFree(GL);
    delete rh, rl, gh, gl, parallel_hm_out, serial_hm_out;

    TIMERSTOP(total_kernel_test);
}


/**
 *
 */
void kernel128_benchmark(){

    cout << "starting kernel 128 benchmark" << endl;

    //set the ID of the CUDA device
    cudaSetDevice(0);   CUERR;
    cudaDeviceReset();  CUERR;
    printDeviceInfo();  CUERR;

    TIMERSTART(mallocHost);
    uint128_t *rh = new uint128_t[NUMINPUTLINES];
    uint128_t *rl = new uint128_t[NUMINPUTLINES];
    uint128_t *gh = new uint128_t[NUMINPUTLINES];
    uint128_t *gl = new uint128_t[NUMINPUTLINES];
    uint128_t *RH = nullptr, *RL = nullptr, *GH = nullptr, *GL = nullptr;
    uint128_t * parallel_errorCount = new uint128_t[NUMINPUTLINES];
    TIMERSTOP(mallocHost);

    TIMERSTART(generateInput);
    generate_random_input(rh, rl, gh, gl);
    TIMERSTOP(generateInput);

    TIMERSTART(cudaMalloc);
    cudaMalloc(&RH, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&RL, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&GH, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    cudaMalloc(&GL, sizeof(uint128_t) * NUMINPUTLINES);    CUERR;
    TIMERSTOP(cudaMalloc);

//    for(int i=32; i <=1024; i+=32) {

//        cout << i << endl;
    TIMERSTART(cudaMemcpyToDevice);
    cudaMemcpy(RH, rh, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(RL, rl, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(GH, gh, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    cudaMemcpy(GL, gl, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyHostToDevice);  CUERR;
    TIMERSTOP(cudaMemcpyToDevice);


    //TIMERSTART(kernels);
    TIMERSTART(CARF128);
    CARF<<<SDIV(NUMINPUTLINES, NUMTHREADS), NUMTHREADS>>>(RH, RL, GH, GL, RL);
    //CARF_loop<<<NUMBLOCKS, NUMTHREADS>>>(RH, RL, GH, GL, RL);
    CUERR;
    TIMERSTOP(CARF128);
    cudaDeviceSynchronize();
    TIMERSTART(parallel_popCount);
    ConservativePopCount<<<SDIV(NUMINPUTLINES, NUMTHREADS), NUMTHREADS>>>(RL, RH);
    //ConservativePopCount_loop<<<NUMBLOCKS, NUMTHREADS>>>(RL, RH);
    CUERR;
    TIMERSTOP(parallel_popCount);
    //TIMERSTOP(kernels);
//    }

    TIMERSTART(cudaMemcpyToHost);
    cudaMemcpy(parallel_errorCount, RH, NUMINPUTLINES * sizeof(uint128_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpyToHost);

    TIMERSTART(countErrors);
    serial_countExceedingThreshold(parallel_errorCount, 2);
    TIMERSTOP(countErrors);

    cudaFree(RH);
    cudaFree(RL);
    cudaFree(GH);
    cudaFree(GL);
    delete rh, rl, gh, gl, parallel_errorCount;
}


///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////


int main(int argc, char * argv[]) {

    cout << "NUMINPUTLINES: " << NUMINPUTLINES << endl << "REPEATS: " << REPEATS << endl << "NUMTHREADS: " << NUMTHREADS
         << endl << "NUMBLOCKS: " << NUMBLOCKS << endl;

//    for(int i = 1; i<5; i++){
//        cout << "Run: " << i << endl;
    //kernel128_test();
    kernel128_benchmark();
//        kernel128_benchmark(1024, 1024);//volta
//    }


//    uint128_t a = 1242;
//    uint128_t b(2323,14);
//    uint128_t d(2323,14);
//    uint128_t c = 1242;
//
//    b.printBits();
//    b=b>>0;
//    cout << endl;
//    b.printBits();
//    cout << endl;
//
//    c.printBits();
//    c=c<<128;
//    c.printBits();
//    cout << endl;
}