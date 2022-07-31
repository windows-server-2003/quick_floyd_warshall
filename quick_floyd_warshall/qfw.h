#pragma once
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <type_traits>
#include <limits>
#include "internal/vectorize.h"

namespace quick_floyd_warshall {

template <class T, class = void> struct is_complete : std::false_type {};
template <class T> struct is_complete<T, decltype(void(sizeof(T)))> : std::true_type {};

template<typename T> struct floyd_warshall_naive {
	using value_t = T;
	static constexpr T INF = std::numeric_limits<T>::max() / 2;
	static std::string get_description() { return "naive<int" + std::to_string(sizeof(value_t) * 8) + "_t>"; }
	static void run(int n, const T *input_matrix, T *output_matrix, bool symmetric = false) {
		(void) symmetric;
		T *buf = (T *) malloc(n * n * sizeof(T));
		memcpy(buf, input_matrix, n * n * sizeof(T));
		for (int k = 0; k < n; k++) for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) 
			buf[i * n + j] = std::min<T>(buf[i * n + j], buf[i * n + k] + buf[k * n + j]);
		memcpy(output_matrix, buf, n * n * sizeof(T));
		free(buf);
	}
};

using InstSet = vectorize::InstSet;

template<InstSet inst_set, typename T, int unroll_type> struct floyd_warshall {
public:
	static constexpr T INF = std::numeric_limits<T>::max() / 2;
	using value_t = T;
	
	static std::string get_description() {
		return "opt<" + vectorize::inst_set_to_str(inst_set) + ", " "int" + std::to_string(sizeof(value_t) * 8) + "_t, "
			+ std::to_string(unroll_type) + ">"; 
	}
private:
	static constexpr int B = 64; // block size
	using vector_t = vectorize::vector_t<inst_set, T>;
	
	static_assert(is_complete<vector_t>::value, "Invalid inst_set or T");
	static_assert(B % (vector_t::SIZE / sizeof(T)) == 0, "Invalid B value");
	static_assert(unroll_type >= 0 && unroll_type <= 3, "Invalid unroll_type value");
	
	/*
		MaxPlusMul?(a, b, c) :
		 - [a, a + B * B), [b, b + B * B), [c, c + B * B) must not overlap
		 - equivalent to:
			for all i, j in [0, B):
				a[i * B + j] = max(a[i * B + j], max{b[i * B + k] + c[k * B + j] | k in [0, B)})
	*/
	static void MaxPlusMul0(T *a, T *b, T *c) {
		constexpr int n = B;
		for (int k = 0; k < n; k += 2) for (int i = 0; i < n; i += 2) {
			vector_t coef00(b[(i + 0) * n + (k + 0)]);
			vector_t coef01(b[(i + 0) * n + (k + 1)]);
			vector_t coef10(b[(i + 1) * n + (k + 0)]);
			vector_t coef11(b[(i + 1) * n + (k + 1)]);
			
			T *aa = a + i * n;
			T *bb = c + k * n;
			for (int j = 0; j < n; j += vector_t::SIZE / sizeof(T)) {
				vector_t t0(bb + j);
				vector_t t1(bb + n + j);
				max(t0 + coef00, t1 + coef01).chmax_store(aa + j);
				max(t0 + coef10, t1 + coef11).chmax_store(aa + n + j);
			}
		}
	}
	static void MaxPlusMul1(T *a, T *b, T *c) {
		constexpr int n = B;
		for (int k = 0; k < n; k += 4) for (int i = 0; i < n; i += 2) {
			vector_t coef00(b[(i + 0) * n + (k + 0)]);
			vector_t coef01(b[(i + 0) * n + (k + 1)]);
			vector_t coef02(b[(i + 0) * n + (k + 2)]);
			vector_t coef03(b[(i + 0) * n + (k + 3)]);
			vector_t coef10(b[(i + 1) * n + (k + 0)]);
			vector_t coef11(b[(i + 1) * n + (k + 1)]);
			vector_t coef12(b[(i + 1) * n + (k + 2)]);
			vector_t coef13(b[(i + 1) * n + (k + 3)]);
			
			T *aa = a + i * n;
			T *bb = c + k * n;
			for (int j = 0; j < n; j += vector_t::SIZE / sizeof(T)) {
				vector_t t0(bb + j);
				vector_t t1(bb + n + j);
				vector_t t2(bb + n + n + j);
				vector_t t3(bb + n + n + n + j);
				max(max(t0 + coef00, t1 + coef01), max(t2 + coef02, t3 + coef03)).chmax_store(aa + j);
				max(max(t0 + coef10, t1 + coef11), max(t2 + coef12, t3 + coef13)).chmax_store(aa + n + j);
			}
		}
	}
	static void MaxPlusMul2(T *a, T *b, T *c) {
		constexpr int n = B;
		for (int k = 0; k < n; k += 2) for (int i = 0; i < n; i += 4) {
			vector_t coef00(b[(i + 0) * n + (k + 0)]);
			vector_t coef01(b[(i + 0) * n + (k + 1)]);
			vector_t coef10(b[(i + 1) * n + (k + 0)]);
			vector_t coef11(b[(i + 1) * n + (k + 1)]);
			vector_t coef20(b[(i + 2) * n + (k + 0)]);
			vector_t coef21(b[(i + 2) * n + (k + 1)]);
			vector_t coef30(b[(i + 3) * n + (k + 0)]);
			vector_t coef31(b[(i + 3) * n + (k + 1)]);
			
			T *aa = a + i * n;
			T *bb = c + k * n;
			for (int j = 0; j < n; j += vector_t::SIZE / sizeof(T)) {
				vector_t t0(bb + j);
				vector_t t1(bb + n + j);
				max(t0 + coef00, t1 + coef01).chmax_store(aa + j);
				max(t0 + coef10, t1 + coef11).chmax_store(aa + n + j);
				max(t0 + coef20, t1 + coef21).chmax_store(aa + n + n + j);
				max(t0 + coef30, t1 + coef31).chmax_store(aa + n + n + n + j);
			}
		}
	}
	static void MaxPlusMul3(T *a, T *b, T *c) {
		constexpr int n = B;
		for (int k = 0; k < n; k += 4) for (int i = 0; i < n; i += 4) {
			vector_t coef00(b[(i + 0) * n + (k + 0)]);
			vector_t coef01(b[(i + 0) * n + (k + 1)]);
			vector_t coef02(b[(i + 0) * n + (k + 2)]);
			vector_t coef03(b[(i + 0) * n + (k + 3)]);
			vector_t coef10(b[(i + 1) * n + (k + 0)]);
			vector_t coef11(b[(i + 1) * n + (k + 1)]);
			vector_t coef12(b[(i + 1) * n + (k + 2)]);
			vector_t coef13(b[(i + 1) * n + (k + 3)]);
			vector_t coef20(b[(i + 2) * n + (k + 0)]);
			vector_t coef21(b[(i + 2) * n + (k + 1)]);
			vector_t coef22(b[(i + 2) * n + (k + 2)]);
			vector_t coef23(b[(i + 2) * n + (k + 3)]);
			vector_t coef30(b[(i + 3) * n + (k + 0)]);
			vector_t coef31(b[(i + 3) * n + (k + 1)]);
			vector_t coef32(b[(i + 3) * n + (k + 2)]);
			vector_t coef33(b[(i + 3) * n + (k + 3)]);
			
			T *aa = a + i * n;
			T *bb = c + k * n;
			for (int j = 0; j < n; j += vector_t::SIZE / sizeof(T)) {
				vector_t t0(bb + j);
				vector_t t1(bb + n + j);
				vector_t t2(bb + n + n + j);
				vector_t t3(bb + n + n + n + j);
				max(max(t0 + coef00, t1 + coef01), max(t2 + coef02, t3 + coef03)).chmax_store(aa + j);
				max(max(t0 + coef10, t1 + coef11), max(t2 + coef12, t3 + coef13)).chmax_store(aa + n + j);
				max(max(t0 + coef20, t1 + coef21), max(t2 + coef22, t3 + coef23)).chmax_store(aa + n + n + j);
				max(max(t0 + coef30, t1 + coef31), max(t2 + coef32, t3 + coef33)).chmax_store(aa + n + n + n + j);
			}
		}
	}
	static void FWI(T *a, T *b, T *c) {
		if (a != b && a != c && b != c) {
			if (unroll_type == 0) MaxPlusMul0(a, b, c);
			if (unroll_type == 1) MaxPlusMul1(a, b, c);
			if (unroll_type == 2) MaxPlusMul2(a, b, c);
			if (unroll_type == 3) MaxPlusMul3(a, b, c);
			return;
		}
		constexpr int n = B;
		for (int k = 0; k < n; k++) for (int i = 0; i < n; i++) {
			vector_t coef(b[i * n + k]);
			
			T *aa = a + i * n;
			T *bb = c + k * n;
			for (int j = 0; j < n; j += vector_t::SIZE / sizeof(T))
				(vector_t(bb + j) + coef).chmax_store(aa + j);
		}
	}
	static void FWR(int n_blocks_power2, int n_blocks, int block_index0, int block_index1, int block_index2,
		T **block_start, bool symmetric) {
		
		if (block_index0 >= n_blocks || block_index1 >= n_blocks || block_index2 >= n_blocks) return;
		if (n_blocks_power2 == 1) {
			FWI(
				block_start[block_index0 * n_blocks + block_index2],
				block_start[block_index0 * n_blocks + block_index1],
				block_start[block_index1 * n_blocks + block_index2]
			);
		} else {
			int half = n_blocks_power2 >> 1;
			if (!symmetric) {
				FWR(half, n_blocks, block_index0       , block_index1       , block_index2       , block_start, false);
				FWR(half, n_blocks, block_index0       , block_index1       , block_index2 + half, block_start, false);
				FWR(half, n_blocks, block_index0 + half, block_index1       , block_index2       , block_start, false);
				FWR(half, n_blocks, block_index0 + half, block_index1       , block_index2 + half, block_start, false);
				FWR(half, n_blocks, block_index0 + half, block_index1 + half, block_index2 + half, block_start, false);
				FWR(half, n_blocks, block_index0 + half, block_index1 + half, block_index2       , block_start, false);
				FWR(half, n_blocks, block_index0       , block_index1 + half, block_index2 + half, block_start, false);
				FWR(half, n_blocks, block_index0       , block_index1 + half, block_index2       , block_start, false);
			} else {
				// if symmetric, block_index0 = block_index1 = block_index2
				FWR(half, n_blocks, block_index0       , block_index1       , block_index2       , block_start, true);
				FWR(half, n_blocks, block_index0       , block_index1       , block_index2 + half, block_start, false);
				transpose_copy(half, n_blocks, block_index0, block_index0 + half, block_start);
				FWR(half, n_blocks, block_index0 + half, block_index1       , block_index2 + half, block_start, false);
				FWR(half, n_blocks, block_index0 + half, block_index1 + half, block_index2 + half, block_start, true);
				FWR(half, n_blocks, block_index0 + half, block_index1 + half, block_index2       , block_start, false);
				transpose_copy(half, n_blocks, block_index0 + half, block_index0, block_start);
				FWR(half, n_blocks, block_index0       , block_index1 + half, block_index2       , block_start, false);
			}
		}
	}
	// copy [block_row_offset:block_row_offset+n)[block_column_offset:block_column_offset+n) to its transposed posititon
	// anything outside n_blocks * n_blocks blocks is ignored
	static void transpose_copy(int n, int n_blocks, int block_row_offset, int block_column_offset, T **block_start) {
		for (int i = block_row_offset; i < block_row_offset + n && i < n_blocks; i++) 
			for (int j = block_column_offset; j < block_column_offset + n && j < n_blocks; j++) {
			
			T *src = block_start[i * n_blocks + j];
			T *dst = block_start[j * n_blocks + i];
			for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) dst[x * B + y] = src[y * B + x];
		}
	}
	/*
		rev == false :
			Copy the n * n elements in src to dst in the order like this(each src[i][j] is a BxB block):
			dst: src[0][0], src[0][1], src[1][0], src[1][1],
				 src[0][2], src[0][3], src[1][2], src[1][3],
				 src[2][0], src[2][1], src[3][0], src[3][1],
				 src[2][2], src[2][3], src[3][2], src[3][3],
				 src[0][4], src[0][5], src[1][4], src[1][5], ...
			and returns the pointer to the next element of the last element written in dst.
			Elements outside the n_blocks * n_blocks blocks will be ignored
			Elements in dst corresponding to elements outside the src_n * src_n but in n_blocks * n_blocks
				will be filled
			block_start[i * n_blocks + j] will point to the starting element in dst corresponding to (i, j) block
		
		rev == true :
			same as rev == false except that the copy direction is reversed and
				elements in dst where INF would be contained if !rev are untouched
		
		This function negates all the element and FWR handles everything with max instead of min.
		This is because chmax(mem, reg) can be implemented faster than chmin(mem, reg) with avx2+int64_t
		The cost of negation should be negligible for other combinations, where this trick is irrelevant
	*/
	static T *reorder(int src_n, int n_blocks_power2, T *dst_head, T *src, T **block_start,
		int block_row, int block_column, bool rev) {
		
		int n_blocks = (src_n + B - 1) / B;
		if (block_row >= n_blocks || block_column >= n_blocks) return dst_head;
		if (n_blocks_power2 == 1) {
			T *src_base = src + (block_row * B * src_n + block_column * B);
			for (int i = 0; i < B; i++) {
				if (block_row * B + i < src_n) {
					int length = std::min(B, src_n - block_column * B);
					if (!rev) {
						for (int j = 0; j < length; j++) dst_head[i * B + j] = -src_base[i * src_n + j];
						for (int j = length; j < B; j++) dst_head[i * B + j] = -INF;
					} else {
						for (int j = 0; j < length; j++) src_base[i * src_n + j] = -dst_head[i * B + j];
					}
				} else {
					if (!rev) std::fill(dst_head + i * B, dst_head + (i + 1) * B, -INF);
				}
			}
			block_start[block_row * n_blocks + block_column] = dst_head;
			return dst_head + B * B;
		} else {
			int n_blocks_p2_half = n_blocks_power2 >> 1;
			// split into 2x2 recursively
			for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
				dst_head = reorder(src_n, n_blocks_p2_half, dst_head, src, block_start,
					block_row + i * n_blocks_p2_half, block_column + j * n_blocks_p2_half, rev);
			return dst_head;
		}
	}
public:
	static void run(int src_n, const T *input_matrix, T *output_matrix, bool symmetric = false) {
		assert(0 <= src_n && src_n < 65536);
		if (src_n == 0) return;
		int n_blocks = (src_n + B - 1) / B; // number of BxB blocks in a row
		int n_blocks_power2 = 1; // smallest power of 2 >= src_n / B
		while (n_blocks_power2 * B < src_n) n_blocks_power2 *= 2;
		
		// allocate and align the needed buffers
		const size_t reordered_needed_size = (B * n_blocks) * (B * n_blocks) * sizeof(T);
		size_t reordered_buffer_size = reordered_needed_size + 64;
		void *reordered_org = malloc(reordered_buffer_size);
		assert(reordered_org);
		void *reordered = reordered_org;
		assert(std::align(64, reordered_needed_size, reordered, reordered_buffer_size));
		// block_start[i][j] : pointer to the starting element of the (i, j) block in t 
		T **block_start = (T **) malloc(n_blocks * n_blocks * sizeof(T *));
		std::fill(block_start, block_start + n_blocks * n_blocks, nullptr);
		
		reorder(src_n, n_blocks_power2, (T *) reordered, const_cast<T *>(input_matrix), block_start, 0, 0, false);
		FWR(n_blocks_power2, n_blocks, 0, 0, 0, block_start, symmetric);
		reorder(src_n, n_blocks_power2, (T *) reordered, output_matrix, block_start, 0, 0, true);
		
		free(reordered_org);
		free(block_start);
	}
};

} // namespace quick_floyd_warshall

