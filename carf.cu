//
// Created by Yorrick on 24.04.2018.
//

/**
 * For the macros to work, you need this hpc_helpers.hpp.
 * I used the implementation found here:
 * https://github.com/JGU-HPC/parallelprogrammingbook/blob/master/include/hpc_helpers.hpp
 */
#include "hpc_helpers.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <string>
#include <iostream>
#include <bitset>
#include <iostream>

using namespace std;

/**
 * Volta:
 * max NUMININPUTLINES: (380895232)
 * ideal NUMTHREADS for both kernels: 1024
 */
 /**
  * Tesla 40c:
  * max NUMINPUTLINES: 397295616UL
  * ideal NUMTHREADS for CARF: 128
  * ideal NUMTHREADS for PopCount: 1024
  *
  */
#define NUMINPUTLINES (1000000)//(304611328UL)//(250000000UL)//(100000000UL)

#define NUMGPUS 2

#define NUMTHREADS (1024)
#define NUMBLOCKS 10000
#define REPEATS (38)

#define PDEL (0.01)
#define PINS (0.01)


///////////////////////////////////////////////////////////////////////////////
//CARF - KERNEL
///////////////////////////////////////////////////////////////////////////////
/**
 * Amends a single Hamming Mask, removing spurious 0s.
 */
__device__ uint64_t SHMS(uint64_t RH, uint64_t RL, uint64_t GH, uint64_t GL){
    return ((((RH ^ GH) | (RL ^ GL))<<1 & ((RH ^ GH) | (RL ^ GL))>>1) | (((RH ^ GH) | (RL ^ GL))<<1 & ((RH ^ GH) | (RL ^ GL))>>2) | (((RH ^ GH) | (RL ^ GL))<<2 & ((RH ^ GH) | (RL ^ GL))>>1) | ((RH ^ GH) | (RL ^ GL)));
}


/**
 * Computes the final bit-vector / Hamming Mask for the given pair using SHMS.
 * Stores final bit-vector in HM_OUT.
 */
__global__ void carf_64(uint64_t * RH, uint64_t * RL, uint64_t * GH, uint64_t * GL, uint64_t * HM_OUT){

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
__global__ void carf_64_loop(uint64_t * RH, uint64_t * RL, uint64_t * GH, uint64_t * GL, uint64_t * HM_OUT){

    const uint64_t thid = (blockDim.x*blockIdx.x + threadIdx.x)*REPEATS;

    if(thid < (NUMINPUTLINES)){

        for(size_t i = 0; i<REPEATS; i++){
            HM_OUT[(thid+i)] = SHMS(RH[(thid+i)], RL[(thid+i)], GH[(thid+i)], GL[(thid+i)]) &
                           SHMS((RH[(thid+i)]>>1), (RL[(thid+i)]>>1), GH[(thid+i)], GL[(thid+i)]) &
                           SHMS((RH[(thid+i)]>>2), (RL[(thid+i)]>>2), GH[(thid+i)], GL[(thid+i)]) &
                           SHMS((RH[(thid+i)]<<1), (RL[(thid+i)]<<1), GH[(thid+i)], GL[(thid+i)]) &
                           SHMS((RH[(thid+i)]<<2), (RL[(thid+i)]<<2), GH[(thid+i)], GL[(thid+i)]);
        }
        //printf("%u\n", HM_OUT[thid]);
    }
}

/**
 * Computes the final bit-vector / Hamming Mask for the given pair using SHMS.
 * Stores final bit-vector in HM_OUT.
 * Intended for use in a multi-GPU or -streams scenario. 
 * Needs the to-be processed batchsize.
 */
__global__ void carf_64(uint64_t * RH, uint64_t * RL, uint64_t * GH, uint64_t * GL, uint64_t * HM_OUT, uint64_t batchsize){

    const size_t thid = blockDim.x*blockIdx.x + threadIdx.x;

    if(thid < batchsize){
        HM_OUT[thid] = SHMS(RH[thid], RL[thid], GH[thid], GL[thid]) &
                       SHMS((RH[thid]>>1), (RL[thid]>>1), GH[thid], GL[thid]) &
                       SHMS((RH[thid]>>2), (RL[thid]>>2), GH[thid], GL[thid]) &
                       SHMS((RH[thid]<<1), (RL[thid]<<1), GH[thid], GL[thid]) &
                       SHMS((RH[thid]<<2), (RL[thid]<<2), GH[thid], GL[thid]);
    }
}
















///////////////////////////////////////////////////////////////////////////////
// Conservative Population Count - KERNEL
///////////////////////////////////////////////////////////////////////////////


/**
 * First and naive implementation of a conservative population count function as described in the SHD paper.
 * Relying exclusively on IF-ELSE-conditions.
 * @Deprecated
 */
__global__ void naive_ConservativePopCount64(uint64_t * hm, size_t * errorCount){

    const auto thid = blockDim.x * blockIdx.x + threadIdx.x;

    if(thid<NUMINPUTLINES){

        uint8_t streak = 0;
        uint8_t totalErrors = 0;

        for(uint8_t i = 0; i < 64; i++){

            if ( ((hm[thid]>>i) & 1) != 0) {
                streak++;
            }
            if (i > 0 and streak > 0 and (((hm[thid]>>i) & 1) == 0) and ((hm[thid]>>(i-1) & 1) != 0)) {
                totalErrors += 1 +((streak + 1) / 3);
                streak = 0;
            } else if(i==63 and streak > 0 and (((hm[thid]>>i) & 1) != 0) and ((hm[thid]>>(i-1) & 1) != 0)){
                totalErrors += 1 +((streak + 1) / 3);
            }
        }
        errorCount[thid] = totalErrors;
    }
}

/**
 * No/less divergence.
 * Replaced IF-ELSE-conditions with computation.
 * Still slower than naive approach.
 * For 300mio about 1265ms.
 *
 * Indeed, not better at all in my tests.
 * @Deprecated
 */
__global__ void BetterConservativePopCount64(uint64_t * hm, size_t * errorCount){

    const auto thid = blockDim.x * blockIdx.x + threadIdx.x;

    if(thid<NUMINPUTLINES){

        uint8_t streak = 0;
        uint8_t totalErrors = 0;

        for(uint8_t i = 0; i < 64; i++){

            streak = streak + (((hm[thid]>>i) & 1) != 0);
            //code * (condition1 + condition2)
            totalErrors += (1 +((streak + 1) / 3)) * (streak > 0 and ((hm[thid]>>(i-1) & 1) != 0) and ((i>0 and (((hm[thid]>>i) & 1) == 0)) or ((i==63) and ((hm[thid]>>i) & 1) != 0)));
            //streak = 0 if condition true
            streak = streak * (!(i > 0 and streak > 0 and (((hm[thid]>>i) & 1) == 0) and ((hm[thid]>>(i-1) & 1) != 0)));
        }
        errorCount[thid] = totalErrors;
    }
}



/**
 * The combination of IF-ELSE-conditions and computation which provided best speed.
 * Use this!
 * Output is stored in errorCount.
 */
__global__ void ConservativePopCount64(uint64_t * hm, size_t * errorCount){

    const auto thid = blockDim.x * blockIdx.x + threadIdx.x;

    if(thid<NUMINPUTLINES){

        uint8_t streak = 0;
        uint8_t totalErrors = 0;

        for(uint8_t i = 0; i < 64; i++){

            streak = streak + (((hm[thid]>>i) & 1) != 0);
            if (i > 0 && streak > 0 and (((hm[thid]>>i) & 1) == 0) and ((hm[thid]>>(i-1) & 1) != 0)) {
                totalErrors += 1 +((streak + 1) / 3);
                streak = 0;
            } else if(i==63 and streak > 0 and (((hm[thid]>>i) & 1) != 0) and ((hm[thid]>>(i-1) & 1) != 0)){
                totalErrors += 1 +((streak + 1) / 3);
            }
        }
        errorCount[thid] = totalErrors;
    }
}

/**
 * Variant for multi-GPU and -streams. 
 * Needs a batchsize!
 */
__global__ void ConservativePopCount64(uint64_t * hm, uint64_t * errorCount, uint64_t batchsize){

    const auto thid = blockDim.x * blockIdx.x + threadIdx.x;

    if(thid<batchsize){

        uint8_t streak = 0;
        uint8_t totalErrors = 0;

        for(uint8_t i = 0; i < 64; i++){

            streak = streak + (((hm[thid]>>i) & 1) != 0);
            if (i > 0 && streak > 0 and (((hm[thid]>>i) & 1) == 0) and ((hm[thid]>>(i-1) & 1) != 0)) {
                totalErrors += 1 +((streak + 1) / 3);
                streak = 0;
            } else if(i==63 and streak > 0 and (((hm[thid]>>i) & 1) != 0) and ((hm[thid]>>(i-1) & 1) != 0)){
                totalErrors += 1 +((streak + 1) / 3);
            }
        }
        errorCount[thid] = totalErrors;
    }
}


/**
 * Allows a single thread to process REPEATS-many pairs.
 */
__global__ void ConservativePopCount64_loop(uint64_t * hm, size_t * errorCount){

    const auto thid = (blockDim.x * blockIdx.x + threadIdx.x)*REPEATS;

    if(thid<NUMINPUTLINES) {

        for (uint16_t j = 0; j < REPEATS; j++) {

            uint8_t streak = 0;
            uint8_t totalErrors = 0;

            for (uint8_t i = 0; i < 64; i++) {

                streak = streak + (((hm[(thid+j)] >> i) & 1) != 0);
                if (i > 0 && streak > 0 and (((hm[(thid+j)] >> i) & 1) == 0) and ((hm[(thid+j)] >> (i - 1) & 1) != 0)) {
                    totalErrors += 1 + ((streak + 1) / 3);
                    streak = 0;
                } else if (i == 63 and streak > 0 and (((hm[(thid+j)] >> i) & 1) != 0) and
                           ((hm[(thid+j)] >> (i - 1) & 1) != 0)) {
                    totalErrors += 1 + ((streak + 1) / 3);
                }
            }
            errorCount[(thid+j)] = totalErrors;
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
 * serial SHMS
 */
uint64_t serialSHMS(uint64_t RH, uint64_t RL, uint64_t GH, uint64_t GL) {
    return ((((RH ^ GH) | (RL ^ GL)) << 1 & ((RH ^ GH) | (RL ^ GL)) >> 1) |
            (((RH ^ GH) | (RL ^ GL)) << 1 & ((RH ^ GH) | (RL ^ GL)) >> 2) |
            (((RH ^ GH) | (RL ^ GL)) << 2 & ((RH ^ GH) | (RL ^ GL)) >> 1) |
            ((RH ^ GH) | (RL ^ GL)));
}

/**
 * serial CARF
 */
void serial_CARF(uint64_t * RH, uint64_t * RL, uint64_t * GH, uint64_t * GL, uint64_t * serial_HM_OUT){


    for(size_t i = 0; i < NUMINPUTLINES; i++){

        serial_HM_OUT[i] = serialSHMS(RH[i], RL[i], GH[i], GL[i]) &
                           serialSHMS((RH[i]>>1), (RL[i]>>1), GH[i], GL[i]) &
                           serialSHMS((RH[i]>>2), (RL[i]>>2), GH[i], GL[i]) &
                           serialSHMS((RH[i]<<1), (RL[i]<<1), GH[i], GL[i]) &
                           serialSHMS((RH[i]<<2), (RL[i]<<2), GH[i], GL[i]);

    }
}

/**
 * Serial and naive conservativer popcount.
 */
void serial_conservative_popcount(uint64_t * final_hm, uint64_t * errorCount) {
    for (size_t index = 0; index < NUMINPUTLINES; index++) {

        std::bitset<64> hm = final_hm[index];
        size_t totalErrors = 0;
        size_t streak = 0;

        for (uint8_t i = 0; i < 64; i++) {
            if (hm[i] == 1) {
                streak++;
            }
            if (i > 0 and hm[i] == 0 and hm[i - 1] == 1) {
                totalErrors += 1 + ((streak + 1) / 3);
                streak = 0;
            } else if (i == 63 and streak > 0 and ((hm[i]) == 1) and ((hm[i - 1] == 1))) {
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
void generate_random_input(uint64_t * rh, uint64_t * rl, uint64_t * gh, uint64_t * gl){

    std::mt19937_64 gen (std::random_device{}());
    std::uint64_t randomNumber = gen();

    for(size_t i = 0; i < NUMINPUTLINES; i++){
        rh[i] = gen();
        rl[i] = gen();
        gh[i] = gen();
        gl[i] = gen();
    }

//    cout << "\nPrinting inputs:" << endl;
//    for (int i = 0; i < 7; i++){
//        std::bitset<64> hr(rh[i]);
//        std::bitset<64> lr(rl[i]);
//        std::bitset<64> hg(gh[i]);
//        std::bitset<64> lg(gl[i]);
//        cout << "rh["<< i <<"] " << hr<< endl;
//        cout << "rl["<< i <<"] " << lr << endl;
//        cout << "gh["<< i <<"] " << hg << endl;
//        cout << "gl["<< i <<"] " << lg << endl;
//    }
//    cout << endl;

}


/**
 * Generates realistic pairs by creating the same word in read and candidate and 
 * changes randomly parts.
 * Probability is given by PDEL and PINS.
 * Very slow in comparison to generate_random_input-function.
 *
 * Used for the experiment.
 */
void levenstein_generator(uint64_t * rh, uint64_t * rl, uint64_t * gh, uint64_t * gl){

    std::mt19937_64 gen (std::random_device{}());

    for(size_t index = 0; index<NUMINPUTLINES; index++){
        uint64_t h = gen();
        uint64_t l = gen();
        std::bitset<64> hr(h);
        std::bitset<64> lr(l);
        std::bitset<64> hg(h);
        std::bitset<64> lg(l);

        uint8_t i = 0, j = 0;

        std::bernoulli_distribution d1(PDEL);
        std::bernoulli_distribution d2(PINS);


        while(i<64 && j<64){
            if(d1(gen)){
                i++;
            }
            if(d2(gen)){
                j++;
            }
            hr[i]=hg[j];
            lr[i]=lg[j];
            i++;
            j++;

        }

        rh[index]=hr.to_ullong();
        rl[index]=lr.to_ullong();
        gh[index]=hg.to_ullong();
        gl[index]=lg.to_ullong();
    }
}

/**
 * Returns and prints number of pairs exceeding error threshold.
 */
size_t serial_countExceedingThreshold(uint64_t * errorCount, uint64_t threshold){
    size_t negatives = 0;

    for(int i = 0; i < NUMINPUTLINES; i++){
        if(errorCount[i]>threshold){
            //cout << errorCount[i] << endl;
            negatives++;
        }
    }

    cout << "Negatives: " << negatives << endl;
    return negatives;
}




///////////////////////////////////////////////////////////////////////////////
// Experiment functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Wagner-Fischer-Algorithm to measure edit distance between two words stored in four bitsets of size 64.
 * @param rh
 * @param rl
 * @param gh
 * @param gl
 * @return levenstein-distance
 */
uint64_t levenstein64(std::bitset<64> rh, std::bitset<64> rl, std::bitset<64> gh, std::bitset<64> gl){
    size_t buf[64+1], best, diag;
    std::iota(buf, buf+64+1, 0);
    for (size_t i = 1; i <= 64; i++) {
        diag = buf[0]++;
        for (size_t j = 1; j <= 64; j++) {
            best=std::min({buf[j]+1, buf[j-1]+1, diag+!(rh[i-1]==gh[j-1] && rl[i-1]==gl[j-1])});
            diag = buf[j];
            buf[j] = best;
        }
    }
    return buf[64];
}

/**
 * Runs WA on arrays of words.
 */
void edit_distance(uint64_t * rh, uint64_t * rl, uint64_t * gh, uint64_t * gl, uint64_t * edit_out){

    for (uint64_t i = 0; i < NUMINPUTLINES; i++){
        edit_out[i] = levenstein64(rh[i], rl[i], gh[i], gl[i]);
    }

}





///////////////////////////////////////////////////////////////////////////////
// Debug functions
///////////////////////////////////////////////////////////////////////////////

/**
 * Compares the final bit-vectors computed by carf and serial carf.
 * Prints differing vectors to console.
 */
void compare_results(uint64_t * serial_hm_out, uint64_t * parallel_hm_out){
    size_t errorcount = 0;
    for(size_t i = 0; i < NUMINPUTLINES; i++){

        if(serial_hm_out[i]!=parallel_hm_out[i]){
            errorcount++;
            cout << "Error in entry " << i << ": serial: " << serial_hm_out[i] << " parallel: " << parallel_hm_out[i] <<endl;
        }
    }
    cout << "Found errors: " << errorcount << endl;
}

/**
 * Decodes uint64_t to human words for easier error search and readability.
 */
void print_words(uint64_t rh, uint64_t rl, uint64_t gh, uint64_t gl){

    char R[64], G[64];
    std::bitset<64>hr(rh);
    std::bitset<64>lr(rl);
    std::bitset<64>hg(gh);
    std::bitset<64>lg(gl);

    for(int i = 0; i<64; i++){
        if(hr[i]==1&&lr[i]==1){
            R[i]='A';
        } else if (hr[i]==0&&lr[i]==1){
            R[i]='C';
        } else if (hr[i]==1&&lr[i]==0){
            R[i]='G';
        } else {
            R[i]='T';
        }
    }

    for(int i = 0; i<64; i++){
        if(hg[i]==1&&lg[i]==1){
            G[i]='A';
        } else if (hg[i]==0&&lg[i]==1){
            G[i]='C';
        } else if (hg[i]==1&&lg[i]==0){
            G[i]='G';
        } else {
            G[i]='T';
        }
    }

    printf("%s \n%s\n", R, G);
}

/**
 * If parallel_edit has more errors than serial_edit, it prints the errorcount and words to console.
 */
void compare_edit_distance(uint64_t * parallel_edit, uint64_t * serial_edit, uint64_t * rh, uint64_t * rl, uint64_t * gh, uint64_t * gl){

    uint64_t error_count = 0;

    for(uint64_t i = 0; i<NUMINPUTLINES; i++){
        if(parallel_edit[i] > serial_edit[i]){

            cout << "Error count: " << parallel_edit[i] << " Edit-distance: " << serial_edit[i] << endl;
            //cout << std::bitset<64>(rh[i]) << " " <<std::bitset<64>(rl[i])<<endl<<std::bitset<64>(gh[i])<<" "<<std::bitset<64>(gl[i])<<endl;
            print_words(rh[i], rl[i], gh[i], gl[i]);

            error_count++;
        }
    }
    cout << "Edit-distance errorcount: " << error_count << endl;
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
 * Test-sequence for words of 64 letters.
 *
 * Computes CARF in serial and in parallel and compares the results against each other.
 *
 * Needs a lot of time if used with NUMINPUTLINES > 10 million (more than 5 minutes).
 * Inputs are generated randomly.
 * To check if correct pairs are processed correctly, the first pair matches.
 * This causes the printed "Negatives = ... " to be at least NUMINPUTLINES - 1.
 *
 */
void kernel_64_test(){

    //set the ID of the CUDA device
    cudaSetDevice(0);   CUERR
    cudaDeviceReset();  CUERR

    printDeviceInfo();

    TIMERSTART(total_kernel64_test);
    cout << "Starting kernel_test-function" << endl;

    uint64_t * rh = new uint64_t[NUMINPUTLINES];
    uint64_t * rl = new uint64_t[NUMINPUTLINES];
    uint64_t * gh = new uint64_t[NUMINPUTLINES];
    uint64_t * gl = new uint64_t[NUMINPUTLINES];
    uint64_t * serial_hm_out = new uint64_t[NUMINPUTLINES];
    uint64_t * parallel_hm_out = new uint64_t[NUMINPUTLINES];

    uint64_t * serial_errorCount = new size_t[NUMINPUTLINES];
    uint64_t * parallel_errorCount = new size_t[NUMINPUTLINES];

    TIMERSTART(generateInput);
    generate_random_input(rh, rl, gh, gl);
    //levenstein_generator(rh, rl, gh, gl);
    TIMERSTOP(generateInput);

    //Set first entries to the same arbitrary value to check if correct pairs are identified.
    rh[0]=rl[0]=gh[0]=gl[0]=10347;

    uint64_t * RH = nullptr, * RL = nullptr, * GH = nullptr, * GL = nullptr;

    TIMERSTART(cudaMalloc);
    cudaMalloc(&RH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&RL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&GH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&GL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;

    TIMERSTOP(cudaMalloc);

    TIMERSTART(cudaMemcpy);
    cudaMemcpy(RH, rh, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
    cudaMemcpy(RL, rl, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
    cudaMemcpy(GH, gh, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
    cudaMemcpy(GL, gl, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
    TIMERSTOP(cudaMemcpy);

    TIMERSTART(kernel);
    carf_64<<<SDIV(NUMINPUTLINES, 1024), 1024>>>( RH, RL, GH, GL, RL);    CUERR;
    TIMERSTOP(kernel);

    TIMERSTART(cudaMemcpy2);
    cudaMemcpy(parallel_hm_out, RL, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpy2);

    TIMERSTART(serial_algorithm);
    serial_CARF(rh, rl, gh, gl, serial_hm_out);
    TIMERSTOP(serial_algorithm);


    TIMERSTART(comparison_hm);
    compare_results(serial_hm_out, parallel_hm_out);
    TIMERSTOP(comparison_hm);

    TIMERSTART(ConservativePopCount64);
    ConservativePopCount64<<<SDIV(NUMINPUTLINES, 1024), 1024>>>(RL, RH); CUERR;
    TIMERSTOP(ConservativePopCount64);

    TIMERSTART(cudaMemcpy4);
    cudaMemcpy(parallel_errorCount, RH, NUMINPUTLINES * sizeof(size_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpy4);

    TIMERSTART(serial_conservativePopC);
    serial_conservative_popcount(serial_hm_out, serial_errorCount);
    TIMERSTOP(serial_conservativePopC);

    TIMERSTART(comparison_errorrate);
    size_t s = serial_countExceedingThreshold(serial_errorCount, 2);
    size_t p = serial_countExceedingThreshold(parallel_errorCount, 2);

    cout << s << "=" << p << endl;

    TIMERSTOP(comparison_errorrate);

    /**
     * Segment for edit distance experiment.
     */
//    TIMERSTART(edit);
//    uint64_t * edit_out = new uint64_t[NUMINPUTLINES];
//    edit_distance(rh, rl, gh, gl, edit_out);
//    compare_edit_distance(parallel_errorCount, edit_out, rh, rl, gh, gl);
//    TIMERSTOP(edit);

    cudaFree(RH);
    cudaFree(RL);
    cudaFree(GH);
    cudaFree(GL);

    delete rh, rl, gh, gl, serial_hm_out, parallel_hm_out, serial_errorCount, parallel_errorCount;

    TIMERSTOP(total_kernel64_test);

}


/**
 * Benchmark-sequence for words of 64 letters.
 *
 * Runs CARF exclusively in parallel to measure the needed time.
 * You must change used kernels manually.
 */
void kernel_64_benchmark(){

    cout << "starting kernel_64_benchmark" << endl;

    //set the ID of the CUDA device
    cudaSetDevice(0);   CUERR;
    cudaDeviceReset();  CUERR;
    printDeviceInfo();  CUERR;

    TIMERSTART(mallocHost);
    uint64_t * rh = new uint64_t[NUMINPUTLINES];
    uint64_t * rl = new uint64_t[NUMINPUTLINES];
    uint64_t * gh = new uint64_t[NUMINPUTLINES];
    uint64_t * gl = new uint64_t[NUMINPUTLINES];
    uint64_t * parallel_errorCount = new uint64_t[NUMINPUTLINES];
    TIMERSTOP(mallocHost);

    TIMERSTART(generateInput);
    generate_random_input(rh, rl, gh, gl);
    //levenstein_generator(rh, rl, gh, gl);
    TIMERSTOP(generateInput);


    TIMERSTART(cudaMalloc);
    uint64_t * RH = nullptr, * RL = nullptr, * GH = nullptr, * GL = nullptr;
    cudaMalloc(&RH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&RL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&GH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    cudaMalloc(&GL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;
    TIMERSTOP(cudaMalloc);

    //for(int i=32; i <=1024; i+=32) { //to detect better parameters
        TIMERSTART(cudaMemcpyToDevice);
        cudaMemcpy(RH, rh, NUMINPUTLINES * sizeof(uint64_t), cudaMemcpyHostToDevice);
        CUERR;
        cudaMemcpy(RL, rl, NUMINPUTLINES * sizeof(uint64_t), cudaMemcpyHostToDevice);
        CUERR;
        cudaMemcpy(GH, gh, NUMINPUTLINES * sizeof(uint64_t), cudaMemcpyHostToDevice);
        CUERR;
        cudaMemcpy(GL, gl, NUMINPUTLINES * sizeof(uint64_t), cudaMemcpyHostToDevice);
        CUERR;
        TIMERSTOP(cudaMemcpyToDevice);



		//cout << i << endl;
		TIMERSTART(carf);
		carf_64<<<SDIV(NUMINPUTLINES, 1024), 1024>>>( RH, RL, GH, GL, RL);    CUERR;
		TIMERSTOP(carf);

        TIMERSTART(parallel_popCount);
		ConservativePopCount64<<<SDIV(NUMINPUTLINES, 1024),1024>>>(RL, RH); CUERR;
        TIMERSTOP(parallel_popCount);
//    }

    TIMERSTART(cudaMemcpyToHost);
    cudaMemcpy(parallel_errorCount, RH, NUMINPUTLINES * sizeof(size_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpyToHost);


    TIMERSTART(countErrors);
    serial_countExceedingThreshold(parallel_errorCount, 2);
    TIMERSTOP(countErrors)

    cudaFree(RH);
    cudaFree(RL);
    cudaFree(GH);
    cudaFree(GL);
    delete rh, rl, gh, gl, parallel_errorCount;

}


/**
 * Test-sequence using 2 GPUs.
 * Results are compared with serial CARF.
 * batchsize = NUMINPUTLINES/2
 */
 //@TODO: Benchmark on symterical system
void twoGPUs(){

    cout << "\nStarting multi-GPU-run with 2 GPUs.\n" << endl;

    uint64_t * rh = nullptr, * rl = nullptr, * gh = nullptr, * gl = nullptr, * parallel_hm_out = nullptr;//, * serial_hm_out = nullptr;


    cudaMallocHost(&rh, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&rl, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&gh, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&gl, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&parallel_hm_out, sizeof(uint64_t)*NUMINPUTLINES);
//    cudaMallocHost(&serial_hm_out, sizeof(uint64_t)*NUMINPUTLINES);

    TIMERSTART(generateInput);
    generate_random_input(rh, rl, gh, gl);
    //generate_better_random_input(rh, rl, gh, gl, 3);

    rh[0]=rl[0]=gh[0]=gl[0]=10347;
    TIMERSTOP(generateInput);

    TIMERSTART(all);

    uint64_t * pointers[10];

//    TIMERSTART(cudaMalloc_1);
    for(int gpu = 0; gpu < 2; gpu++){
        cudaSetDevice(gpu); CUERR;

        TIMERSTART(malloc);
        uint64_t * RH = nullptr, * RL = nullptr, * GH = nullptr, * GL = nullptr, * HM_OUT = nullptr;
        cudaMalloc(&RH, sizeof(uint64_t)*(NUMINPUTLINES/2));   CUERR;
        cudaMalloc(&RL, sizeof(uint64_t)*(NUMINPUTLINES/2));   CUERR;
        cudaMalloc(&GH, sizeof(uint64_t)*(NUMINPUTLINES/2));   CUERR;
        cudaMalloc(&GL, sizeof(uint64_t)*(NUMINPUTLINES/2));   CUERR;
        cudaMalloc(&HM_OUT, sizeof(uint64_t)*(NUMINPUTLINES/2));   CUERR;

        pointers[0+gpu*5]=RH; CUERR;
        pointers[1+gpu*5]=RL; CUERR;
        pointers[2+gpu*5]=GH; CUERR;
        pointers[3+gpu*5]=GL; CUERR;
        pointers[4+gpu*5]=HM_OUT; CUERR;

        TIMERSTOP(malloc);

        //printf("Malloc in Device %d finished.\n", gpu);
    }

//    TIMERSTOP(cudaMalloc_1); CUERR;

//    TIMERSTART(cudaMemcpy_kernel); CUERR;

    for(int gpu = 0; gpu < 2; gpu++){
        cudaSetDevice(gpu); CUERR;
        const int offset = ((NUMINPUTLINES/2)*gpu);
        //printf("Starting memcpy on device %d\n", gpu);

        TIMERSTART(memcpy);
        cudaMemcpyAsync(pointers[0+gpu*5], rh+offset, (NUMINPUTLINES/2)*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
        cudaMemcpyAsync(pointers[1+gpu*5], rl+offset, (NUMINPUTLINES/2)*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
        cudaMemcpyAsync(pointers[2+gpu*5], gh+offset, (NUMINPUTLINES/2)*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
        cudaMemcpyAsync(pointers[3+gpu*5], gl+offset, (NUMINPUTLINES/2)*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
        TIMERSTOP(memcpy);

        TIMERSTART(kernel);
        //printf("Starting kernel on device %d\n", gpu);
        carf_64<<<SDIV((NUMINPUTLINES/2), NUMTHREADS), NUMTHREADS>>>( pointers[0+gpu*5], pointers[1+gpu*5], pointers[2+gpu*5], pointers[3+gpu*5], pointers[4+gpu*5], (NUMINPUTLINES/2));    CUERR;
        TIMERSTOP(kernel);
    }

//    TIMERSTOP(cudaMemcpy_kernel);

    for(int gpu = 0; gpu < 2; gpu++){
        cudaSetDevice(gpu); CUERR;
        ConservativePopCount64<<<SDIV((NUMINPUTLINES/2), NUMTHREADS), NUMTHREADS>>>(pointers[4+gpu*5], pointers[0+gpu*5], (NUMINPUTLINES/2)); CUERR;
        cudaMemcpyAsync(parallel_hm_out+((NUMINPUTLINES/2)*gpu), pointers[0+gpu*5], (NUMINPUTLINES/2)*sizeof(uint64_t), cudaMemcpyDeviceToHost); CUERR;
    }

    for(int gpu = 0; gpu < 2; gpu++){
        cudaSetDevice(gpu);
        cudaFree(pointers[0+gpu*5]); CUERR;
        cudaFree(pointers[1+gpu*5]); CUERR;
        cudaFree(pointers[2+gpu*5]); CUERR;
        cudaFree(pointers[3+gpu*5]); CUERR;
        cudaFree(pointers[4+gpu*5]); CUERR;
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    TIMERSTOP(all);


//    TIMERSTART(serial_algorithm);
//    serial_CARF(rh, rl, gh, gl, serial_hm_out);
//    TIMERSTOP(serial_algorithm);
//
//    TIMERSTART(comparison);
//    compare_results(serial_hm_out, parallel_hm_out);
//    TIMERSTOP(comparison);

     size_t p = serial_countExceedingThreshold(parallel_hm_out, 2);

     cudaFreeHost(rh);
     cudaFreeHost(rl);
     cudaFreeHost(gh);
     cudaFreeHost(gl);
     cudaFreeHost(parallel_hm_out);
//    cudaFreeHost(serial_hm_out);

}



/**
 * WIP
 *
 * Implementing multiple streams.
 * Erroneous behavior, debuging in progress, must be fixed, major rework needed.
 * @TODO: Reimplement streams for both kernels and find errors.
 * @Deprecated
 */
void multiStreams(){

    uint64_t * rh = nullptr, * rl = nullptr, * gh = nullptr, * gl = nullptr, * parallel_hm_out = nullptr;


    cudaMallocHost(&rh, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&rl, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&gh, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&gl, sizeof(uint64_t)*NUMINPUTLINES);
    cudaMallocHost(&parallel_hm_out, sizeof(uint64_t)*NUMINPUTLINES);


    uint64_t * serial_hm_out = new uint64_t[NUMINPUTLINES];


    TIMERSTART(generateInput);
    generate_random_input(rh, rl, gh, gl);
    TIMERSTOP(generateInput);

    rh[0]=rl[0]=gh[0]=gl[0]=4321098;

    uint64_t * RH = nullptr, * RL = nullptr, * GH = nullptr, * GL = nullptr, * HM_OUT = nullptr;

    TIMERSTART(cudaMalloc);
    cudaMalloc(&RH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR ;
    cudaMalloc(&RL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR ;
    cudaMalloc(&GH, sizeof(uint64_t)*NUMINPUTLINES);   CUERR ;
    cudaMalloc(&GL, sizeof(uint64_t)*NUMINPUTLINES);   CUERR ;
    cudaMalloc(&HM_OUT, sizeof(uint64_t)*NUMINPUTLINES);   CUERR;

    TIMERSTOP(cudaMalloc);

    TIMERSTART(cudaMemcpy);
    cudaMemcpy(RH, rh, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR ;
    cudaMemcpy(RL, rl, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR ;
    cudaMemcpy(GH, gh, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR ;
    cudaMemcpy(GL, gl, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyHostToDevice); CUERR;
    TIMERSTOP(cudaMemcpy);

    TIMERSTART(kernel);
    carf_64<<<SDIV(NUMINPUTLINES, NUMTHREADS), NUMTHREADS>>>( RH, RL, GH, GL, HM_OUT);    CUERR
    TIMERSTOP(kernel);

    cudaDeviceSynchronize();

    TIMERSTART(cudaMemcpy2);
    cudaMemcpy(parallel_hm_out, HM_OUT, NUMINPUTLINES*sizeof(uint64_t), cudaMemcpyDeviceToHost); CUERR;
    TIMERSTOP(cudaMemcpy2);

    TIMERSTART(serial_algorithm);
    serial_CARF(rh, rl, gh, gl, serial_hm_out);
    TIMERSTOP(serial_algorithm);


    TIMERSTART(comparison_hm);
    compare_results(serial_hm_out, parallel_hm_out);
    TIMERSTOP(comparison_hm);


    uint64_t * serial_errorCount = new uint64_t[NUMINPUTLINES];
    uint64_t * parallel_errorCount = new uint64_t[NUMINPUTLINES];

    //create streams
    const uint64_t numstreams = 1;//replace with define
    const uint64_t batchsize = (NUMINPUTLINES/numstreams);
    cudaStream_t streams[numstreams]; CUERR;
    for (uint64_t streamID = 0; streamID < numstreams; streamID++){
        cudaStreamCreate(&streams[streamID]); CUERR;
    }

    TIMERSTART(parallel_popCount_streams);
    for(uint64_t streamID = 0; streamID < numstreams; streamID++){
        const uint64_t offset = streamID*batchsize;
        //naive_ConservativePopCount64_streams<<<SDIV(NUMINPUTLINES, NUMTHREADS), NUMTHREADS, streams[streamID]>>>(HM_OUT+offset, RH+offset, batchsize); CUERR; //erroneous part?
        cudaMemcpyAsync(parallel_errorCount+offset, RH+offset, batchsize * sizeof(size_t), cudaMemcpyDeviceToHost, streams[streamID]); CUERR;
    }
    TIMERSTOP(parallel_popCount_streams);

    TIMERSTART(serial_conservativePopC);
    //serial_conservative_popcount(serial_hm_out, serial_errorCount);
    TIMERSTOP(serial_conservativePopC);

    TIMERSTART(comparison_errorrate);
    //size_t s = serial_countExceedingThreshold(serial_errorCount, 2);
    size_t p = serial_countExceedingThreshold(parallel_errorCount, 2);

    //cout << s << "=" << p << endl;

    TIMERSTOP(comparison_errorrate)

}



///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char * argv[]){

    printf("\nStarted main\n");

    cout << "NUMINPUTLINES: " << NUMINPUTLINES << endl;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("Number of devices: %d\n", count);

    cout << "Testing" << endl;
//    cout << "REPEATS: " << REPEATS << endl << "NUMTHREADS: " << NUMTHREADS << endl << "NUMBLOCKS: " << NUMBLOCKS << endl;
    kernel_64_test();
//
    cout << endl << "Start benchmark" << endl;
//
//    for(int i = 1; i<5; i++){
//        cout << endl << "Run " << i << ":" << endl;
        kernel_64_benchmark();
//    }

    twoGPUs();

    //multiStreams(); //broken

    printf("\nmain finished\n");



}
