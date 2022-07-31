## quick_floyd_warshall
A highly-tuned C++ implementation of the Floyd-Warshall algorithm for the APSP(All Pairs Shortest Path) problem.  
This library is targeted to modern(roughly after 2010) x86-64 machines.  
A major part of the speedup is based on the paper below:  
J. S. Park, M. Penner and V. K. Prasanna, "Optimizing graph algorithms for improved cache performance," in IEEE Transactions on Parallel and Distributed Systems, vol. 15, no. 9, pp. 769-782, Sept. 2004, doi: 10.1109/TPDS.2004.44.

## Usage
Include `quick_floyd_warshall/qfw.h` and write
```
quick_floyd_warshall::floyd_warshall<InstSet::AVX2, int64_t, 0>::run(n, matrix, matrix);
```
when `n` is the number of vertices in the graph and `matrix` is a `n * n` array of `int64_t` where `matrix[i * n + j]` is the weight of the edge connecting vertex i and vertex j.  
If there is no edge, you should put `quick_floyd_warshall::floyd_warshall<InstSet::AVX2, int64_t, 0>::INF` instead.  
You can change `AVX2` to `DEFAULT`(no SIMD), `SSE4_2` (SSE4.2), or `AVX512` according to what SIMD instruction set you want to use.  
You have to add the corresponding target options(`-msse4.2`, `-mavx2`, `-mavx512f`, and/or `-mavx512bw`) when compiling your code.  
You can also change `int64_t` to `int16_t` or `int32_t`. Choose this type so that (n - 1) \* max{abs(weight)} is less than half of the maximum value in the type.  
The last template parameter is `unroll_type` and should be an integer between 0 and 3. This affects the performance, but it depends on other parameters and the environment which one is the fastet.  
Negative edge cost is allowed, but **negative cycle is not yet supported**.  
For more detailed specification, see document.md.  

## Example benchmarks
Results of benchmarks on my PC as a reference(conditions below)

 - Intel Core i5-10600k
 - Arch Linux live
 - GCC 12.1.0
 - `-O3 -funroll-loops`
 - symmetric = false
 - average of 5 runs

#### T = int64_t

| N    |  Naive  | inst_set = DEFAULT, unroll_type = 0 | inst_set = AVX2, unroll_type = 3 |
| ---- | ------- | ----------------------------------- | -------------------------------- |
| 512  | 54.6 ms | 43.3 ms                             | 10.7 ms                          |
| 1024 | 469 ms  | 296 ms                              | 86.9 ms                          |
| 1536 | 2190 ms | 932 ms                              | 294 ms                           |
| 2048 | 5320 ms | 2127 ms                             | 697 ms                           |

#### T = int32_t

| N    |  Naive  | inst_set = DEFAULT, unroll_type = 0 | inst_set = AVX2, unroll_type = 3 |
| ---- | ------- | ----------------------------------- | -------------------------------- |
| 512  | 45.3 ms | 42.5 ms                             | 3.17 ms                          |
| 1024 | 368 ms  | 290 ms                              | 24.0 ms                          |
| 1536 | 1300 ms | 907 ms                              | 79.4 ms                          |
| 2048 | 3630 ms | 2060 ms                             | 186 ms                           |


## License
You can use the code under the terms of the GNU General Public License GPL v3 or under the terms of any later revisions of the GPL. Refer to the provided LICENSE file for further information.
