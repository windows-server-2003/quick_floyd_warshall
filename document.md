## Overview
```
namespace quick_floyd_warshall {
	enum class InstSet {
		DEFAULT,
		SSE4_2,
		AVX2,
		AVX512
	};
	template<InstSet inst_set, typename T, int unroll_type> struct floyd_warshall {
		typename value_t;
		static constexpr value_t INF;
		static void run(int src_n, const value_t *input_matrix, value_t *output_matrix, bool symmetric = false);
	}
	template<typename T> struct floyd_warshall_naive {
		typename value_t;
		static constexpr value_t INF;
		static void run(int src_n, const value_t *input_matrix, value_t *output_matrix, bool symmetric = false);
	}
}
```

### InstSet
- DEFAULT : No vectorization, use pure x86_64 instructions
- SSE4_2 : use up to SSE4.2
- AVX2 : use up to AVX2
- AVX512 : use up to AVX-512; requires AVX512BW support in addition to AVX512F when combining with T = int16_t

### struct floyd_warshall
- Template parameters
	- inst_set : vectorization instruction set to be used; must be one of the choices in InstSet
	- T : the type of distance; must be int16_t, int32_t, or int64_t
	- unroll_type : must be one of 0, 1, 2, 3 and affects the performance;  
		It depends on inst_set, T, and CPU architecture which one is the fastest
	
	Violation of the constraints on template parameters results in a static_assert failure

- Members
	 - `value_t` : the same type as `T`
	 - `INF` : equals `std::numeric_limits<T>::max() / 2`; see below for the meaning and usage of this value  
	 - `run(src_n, input_matrix, output_matrix)`
		 - `src_n` : the number of vertices in the graph; must be between 0 and 65535
		 - `input_matrix` : adjacent matrix of the input graph, with `input_matrix[i * src_n + j]` corresponding to the weight of the edge betwenn vertex `i` and `j`.  
			`INF` indicates there is no edge.  
			Any path in the input graph must have a total weight with absolute value less than INF.  
			The graph must not contain negative cycles.  
			`input_matrix[i * src_n + i]` must be zero for all $0 \le \mathrm{i} \lt \mathrm{src\\_n}$  
		 - `output_matrix` : the pointer to which the resulting distance matrix will be written, with similar index correspondence as `input_matrix`  
			It must have the space for `src_n * src_n` `value_t` elements and may overlap with input_matrix.  
			`INF` will be written if the corresponding vertices are disconnected in the graph.  
		 - `symmetric` : can be `true` when the input matrix is symmetric(i.e. all the edges are undirected).  
			This reduces the running time to approximately $\frac{2}{3}$ times of the original time.  



