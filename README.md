# CARF

This project is my bachelors thesis.
CARF is a CUDA Accelerated Read Filter, combining the SHD- and PUNAS-algorithm 
into a single algorithm aimed torwards NVIDIA's CUDA-platform.

It's peak performance is reached at 3,4 billion read-candidate-pairs of length 
64 processed per second.

The error threshold is fixed on e = 2.

Pairs musst be encoded as four integers: Read-High and Read-Low for Reads and 
Genome-High and Genome-Low for candidateregion. Only four letters alphabets are supported.

Implementations for words of 64 and 128 letters are provided through uint64\_t and a custom uint128\_t.
You should use a better performing uint128_t datatype than mine.
Support for longer words is easy to add through using a "longer" datatype and by 
replacing all uints in the original kernels with your type.

CARF itself does not care about the semantics it processes, as long as it is encoded as bitwise.

Compile carf.cu for words of 64 letters (or carf128.cu respectively).
To compile I used "nvcc carf.cu -std=c++14 -o carf".

The provided uint128_t datatype only supports the functions required for CARF to function properly and should not be used as integer outside this project.

For the macros to work you need this hpc_helpers.hpp.
I used the implementation found here: https://github.com/JGU-HPC/parallelprogrammingbook/blob/master/include/hpc_helpers.hpp