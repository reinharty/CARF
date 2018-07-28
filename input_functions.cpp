//
// Created by Yorrick on 21.05.2018.
//


#include <string>
#include <vector>
#include <sstream> //istringstream
#include <iostream> // cout
#include <fstream> // ifstream
#include <cstring>
#include <bitset>

using namespace std;


/**
 * For words of length 64.
 *
 * Encodes lowercase pairs from .csv to vectors rh, rl, gh, gl.
 * Use Vector.data() to access internal arrays.
 *
 * lengths and positions store additional data but are not needed right now.
 *
 * @param inputFileName
 * @param gh
 * @param gl
 * @param rh
 * @param rl
 * @param lengths
 * @param positions
 */
void parseCSVtoBV(string inputFileName, vector<uint64_t>& gh, vector<uint64_t> &gl, vector<uint64_t>&rh,
                  vector<uint64_t>&rl, vector<size_t>&lengths, vector<size_t>&positions){
    ifstream in(inputFileName);
    string buf;

    printf("Started parse\n");
    while (getline(in, buf)) {
        //printf("loop 1\n");

        auto iter = buf.begin();
        uint64_t gh_buf{0}, gl_buf{0}, rh_buf{0}, rl_buf{0};
        size_t count = 0;

        while(*iter!=',') {
            //for(int i = 0; i <64; i++){
            //printf("loop 2");
            count++;
            switch (*iter) {
                case 'g':
                    rh_buf++;
                case 'c':
                    rl_buf++;
                    break;
                case 't':
                    rh_buf++;
            }
            if (count%64==0) {
                rh.push_back(rh_buf);
                rl.push_back(rl_buf);
            }
            rh_buf<<=1;
            rl_buf<<=1;


            *iter++;
        }
        lengths.push_back(count);
        positions.push_back(count/8);
        while(count++%64!=0) {
            rh_buf<<=1;
            rl_buf<<=1;
        }

        *iter++;

        count = 0;
        while(*iter!='\0') {
            //printf("loop 3 ");
            count++;
            switch (*iter) {
                case 'g':
                    gh_buf++;
                case 'c':
                    gl_buf++;
                    break;
                case 't':
                    gh_buf++;
            }

            if (count%64==0) {
                gh.push_back(gh_buf);
                gl.push_back(gl_buf);
            }
            gh_buf<<=1;
            gl_buf<<=1;


            *iter++;
        }
        while(count++%64!=0) {
            gh_buf<<=1;
            gl_buf<<=1;
        }
    }

    cout << "\n" << gh.size() << " " << gl.size() << " " << rh.size() << " " << rl.size() << endl;

    /**cout << "\nPrinting inputs:" << endl;
    for (int i = 0; i < rh.size(); i++){
        std::bitset<64> hr(rh[i]);
        std::bitset<64> lr(rl[i]);
        std::bitset<64> hg(gh[i]);
        std::bitset<64> lg(gl[i]);
        cout << "rh["<< i <<"] " << hr<< endl;
        cout << "rl["<< i <<"] " << lr << endl;
        cout << "gh["<< i <<"] " << hg << endl;
        cout << "gl["<< i <<"] " << lg << endl;
    }
    cout << endl;*/


}


/**
 *
 * Use Vector.size() as dynamic replacement for NUMINPUTLINES.
 *
 * Example sequence using this function can start like this:
 *
    std::vector<uint64_t> gh, gl, rh, rl;
    std::vector<size_t>lengths, positions;

    TIMERSTART(read_from_disk)
    parseCSVtoBV("test.csv", gh, gl, rh, rl, lengths, positions);
    TIMERSTOP(read_from_disk)


    size_t vectorsize=gh.size();

    cout << "vectorsize: " << vectorsize << endl;
    uint64_t * hm_out = new uint64_t[vectorsize];


//    //set the ID of the CUDA device
    cudaSetDevice(0);   CUERR
    cudaDeviceReset();  CUERR

    //allocate storage on GPU
    uint64_t * RH = nullptr, * RL = nullptr, * GH = nullptr, * GL = nullptr, * HM_OUT = nullptr;

    printf("\nAllocating memory");

    TIMERSTART(cudaMalloc)
    cudaMalloc(&RH, sizeof(uint64_t)*vectorsize);   CUERR
    cudaMalloc(&RL, sizeof(uint64_t)*vectorsize);   CUERR
    cudaMalloc(&GH, sizeof(uint64_t)*vectorsize);   CUERR
    cudaMalloc(&GL, sizeof(uint64_t)*vectorsize);   CUERR
    cudaMalloc(&HM_OUT, sizeof(uint64_t)*vectorsize);   CUERR

    TIMERSTOP(cudaMalloc)

    and so forth.
    */