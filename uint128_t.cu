// BACHELOR ARBEIT ABGABE
//
// Created by Yorrick on 22.05.2018.
//
// This is a PSEUDO uint128_t!
// It won't work as a correct int or decimal representation as I only care about the binary representation in the CARF context.
//
// For the version used to create the published benchmarks, checkout original commit.
#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
#include <cstring>
#include <bitset>

using namespace std;


struct uint128_t {

    uint64_t LEFT;
    uint64_t RIGHT;

    __host__ __device__ uint128_t() {
        LEFT = 0;
        RIGHT = 0;
    }

    __host__ __device__ uint128_t(uint64_t right) {
        LEFT = 0;
        RIGHT = right;
    }

    __host__ __device__ uint128_t(uint64_t left, uint64_t right) {
        LEFT = left;
        RIGHT = right;
    }

    /**
     * First version, used in the published benchmarks.
     * a = a>>3;
     */
    __host__ __device__ uint128_t operator<<(const size_t &n) const {
        if(n >= 128){
            return uint128_t{0, 0};
        }

        if(n > 64){
            return uint128_t{(RIGHT<<(n-64)), 0};
        }

        if(n == 64){
            return uint128_t{RIGHT, 0};
        }
        return uint128_t{(LEFT << n) | (RIGHT >> (64 - n)), RIGHT << n};
    }


    /**
     * First version, used in the published benchmarks.
     */
    __host__ __device__ uint128_t operator>>(const size_t &n) const {
        if(n >= 128){
            return uint128_t{0, 0};
        }

        if(n > 64){
            return uint128_t{0, (LEFT>>(n-64))};
        }

        if(n == 64){
            return uint128_t{0, LEFT};
        }

        return uint128_t{(LEFT >> n), (LEFT << (64 - n) | (RIGHT >> n))};
    }

    __host__ __device__ uint128_t operator|(const uint128_t &n) const {
        return uint128_t{(n.LEFT | LEFT), (n.RIGHT | RIGHT)};
    }

    __host__ __device__ uint128_t operator&(const uint128_t &n) const {
	    return uint128_t{(n.LEFT & LEFT), (n.RIGHT & RIGHT)};
    }

    __host__ __device__ uint128_t operator^(const uint128_t &n) const {
        return uint128_t{(n.LEFT^LEFT),(n.RIGHT^RIGHT)};
    }

    __host__ __device__ bool operator!=(const uint128_t &n) const {
        return !(LEFT==n.LEFT and RIGHT==n.RIGHT);
    }

    __host__ __device__ bool operator==(const uint64_t &n) const {
        return (LEFT == 0 and RIGHT == n);
    }

    __host__ __device__ bool operator==(const uint128_t &n) const {
        return (LEFT == n.LEFT and RIGHT == n.RIGHT);
    }

    __host__ __device__ bool operator<(const uint64_t &n) const {
        return(LEFT > 0 or RIGHT < n);
    }

    __host__ __device__ bool operator>(const uint64_t &n) const {
        return (LEFT > 0 or RIGHT > n);
    }

    /**
    bitArray getBits(){
        transform in bitrepresentation
    */

    __device__ void printC(){
        printf("%llu , %llu \n", LEFT, RIGHT);
    }

    __host__ void print() {
        if(LEFT==0){
            cout << RIGHT;
        } else {
            cout << LEFT << "," << RIGHT;
        }
    }

    __host__ void printBits(){
        std::bitset<64> le = LEFT;
        std::bitset<64> ri = RIGHT;
        cout << le <<","<< ri << endl;
    }

};
