#pragma once
#include <cstdint>
#include <type_traits>
#include <string>
#include <algorithm>
#include <immintrin.h>

namespace quick_floyd_warshall {
namespace vectorize {

// instruction set for vectorization
enum class InstSet {
	DEFAULT,
	SSE4_2,
	AVX2,
	AVX512
};
std::string inst_set_to_str(InstSet inst_set) {
	if (inst_set == InstSet::DEFAULT) return "DEFAULT";
	if (inst_set == InstSet::SSE4_2 ) return "SSE4_2";
	if (inst_set == InstSet::AVX2   ) return "AVX2";
	if (inst_set == InstSet::AVX512 ) return "AVX512";
	return "";
}

// wrapper of sse/avx intrinsics
template<InstSet inst_set> class vector_base_t;
template<> class vector_base_t<InstSet::SSE4_2> {
public:
	static constexpr int SIZE = 16;
	using internal_vector_t = __m128i;
	internal_vector_t vec;
	
	vector_base_t ()  = default;
	vector_base_t (internal_vector_t vec_) : vec(vec_) {}
	vector_base_t (void *ptr) : vec(_mm_load_si128((internal_vector_t *) ptr)) {}
	vector_base_t &store(void *ptr) { _mm_store_si128((internal_vector_t *) ptr, vec); return *this; }
};
template<InstSet inst_set> class vector_base_t;
template<> class vector_base_t<InstSet::AVX2> {
public:
	static constexpr int SIZE = 32;
	using internal_vector_t = __m256i;
	internal_vector_t vec;
	
	vector_base_t ()  = default;
	vector_base_t (internal_vector_t vec_) : vec(vec_) {}
	vector_base_t (void *ptr) : vec(_mm256_load_si256((internal_vector_t *) ptr)) {}
	vector_base_t &store(void *ptr) { _mm256_store_si256((internal_vector_t *) ptr, vec); return *this; }
};
template<> class vector_base_t<InstSet::AVX512> {
public:
	static constexpr int SIZE = 64;
	using internal_vector_t = __m512i;
	internal_vector_t vec;
	
	vector_base_t ()  = default;
	vector_base_t (internal_vector_t vec_) : vec(vec_) {}
	vector_base_t (void *ptr) : vec(_mm512_load_si512((internal_vector_t *) ptr)) {}
	vector_base_t &store(void *ptr) { _mm512_store_si512((internal_vector_t *) ptr, vec); return *this; }
};

template<InstSet inst_set, typename T> class vector_t;

/*
	vec.chmin_store(mem): mem[i] = min(mem[i], vec[i])
	vec.chmax_store(mem): mem[i] = max(mem[i], vec[i])
*/


// DEFAULT / *
template<typename T> class vector_t<InstSet::DEFAULT, T> {
	static_assert(std::is_same<T, int16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value, "");
public:
	static constexpr int SIZE = sizeof(T);
	T val;
	vector_t &store(void *ptr) { *((T *) ptr) = val; return *this; }
	vector_t (void *val) : val(*((T *)val)) {}
	vector_t (T val) : val(val) {}
	vector_t operator + (const vector_t &rhs) const { return { T(val + rhs.val) }; }
	vector_t operator - (const vector_t &rhs) const { return { T(val - rhs.val) }; }
	vector_t operator - () const { return { -val }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { std::min(lhs.val, rhs.val) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { std::max(lhs.val, rhs.val) }; }
	vector_t &chmin_store(void *ptr) { if (*((T *) ptr) > val) store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { if (*((T *) ptr) < val) store(ptr); return *this; }
};



// SSE4.2 / int16_t
template<> class vector_t<InstSet::SSE4_2, int16_t> : public vector_base_t<InstSet::SSE4_2> {
public:
	using vector_base_t<InstSet::SSE4_2>::vector_base_t;
	vector_t (int16_t val) : vector_base_t(_mm_set1_epi16(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm_add_epi16(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm_sub_epi16(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm_sub_epi16(_mm_setzero_si128(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm_min_epi16(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm_max_epi16(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) { min(*this, vector_t(ptr)).store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { max(*this, vector_t(ptr)).store(ptr); return *this; }
};
// SSE4.2 / int32_t
template<> class vector_t<InstSet::SSE4_2, int32_t> : public vector_base_t<InstSet::SSE4_2> {
public:
	using vector_base_t<InstSet::SSE4_2>::vector_base_t;
	vector_t (int32_t val) : vector_base_t(_mm_set1_epi32(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm_add_epi32(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm_sub_epi32(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm_sub_epi32(_mm_setzero_si128(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm_min_epi32(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm_max_epi32(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) { min(*this, vector_t(ptr)).store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { max(*this, vector_t(ptr)).store(ptr); return *this; }
};
// SSE4.2 / int64_t
template<> class vector_t<InstSet::SSE4_2, int64_t> : public vector_base_t<InstSet::SSE4_2> {
public:
	using vector_base_t<InstSet::SSE4_2>::vector_base_t;
	vector_t (int64_t val) : vector_base_t(_mm_set1_epi64((__m64) val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm_add_epi64(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm_sub_epi64(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm_sub_epi64(_mm_setzero_si128(), vec) }; }
	// SSE4 doesn't have _mm_min_epi64 / _mm_max_epi64
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return {
		_mm_blendv_epi8(lhs.vec, rhs.vec, _mm_cmpgt_epi64(lhs.vec, rhs.vec))
	}; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return {
		_mm_blendv_epi8(lhs.vec, rhs.vec, _mm_cmpgt_epi64(rhs.vec, lhs.vec))
	}; }
	vector_t &chmin_store(void *ptr) { min(*this, vector_t(ptr)).store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { max(*this, vector_t(ptr)).store(ptr); return *this; }
};



// AVX2 / int16_t
template<> class vector_t<InstSet::AVX2, int16_t> : public vector_base_t<InstSet::AVX2> {
public:
	using vector_base_t<InstSet::AVX2>::vector_base_t;
	vector_t (int16_t val) : vector_base_t(_mm256_set1_epi16(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm256_add_epi16(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm256_sub_epi16(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm256_sub_epi16(_mm256_setzero_si256(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm256_min_epi16(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm256_max_epi16(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) { min(*this, vector_t(ptr)).store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { max(*this, vector_t(ptr)).store(ptr); return *this; }
};
// AVX2 / int32_t
template<> class vector_t<InstSet::AVX2, int32_t> : public vector_base_t<InstSet::AVX2> {
public:
	using vector_base_t<InstSet::AVX2>::vector_base_t;
	vector_t (int32_t val) : vector_base_t(_mm256_set1_epi32(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm256_add_epi32(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm256_sub_epi32(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm256_sub_epi32(_mm256_setzero_si256(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm256_min_epi32(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm256_max_epi32(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) { min(*this, vector_t(ptr)).store(ptr); return *this; }
	vector_t &chmax_store(void *ptr) { max(*this, vector_t(ptr)).store(ptr); return *this; }
};
// AVX2 / int64_t
template<> class vector_t<InstSet::AVX2, int64_t> : public vector_base_t<InstSet::AVX2> {
public:
	using vector_base_t<InstSet::AVX2>::vector_base_t;
	vector_t (int64_t val) : vector_base_t(_mm256_set1_epi64x(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm256_add_epi64(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm256_sub_epi64(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm256_sub_epi64(_mm256_setzero_si256(), vec) }; }
	// avx2 doesn't have _mm256_min_epi64 / _mm256_max_epi64
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return {
		_mm256_blendv_epi8(lhs.vec, rhs.vec, _mm256_cmpgt_epi64(lhs.vec, rhs.vec))
	}; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return {
		_mm256_blendv_epi8(lhs.vec, rhs.vec, _mm256_cmpgt_epi64(rhs.vec, lhs.vec))
	}; }
	vector_t &chmin_store(void *ptr) { // slower because of separate load instruction in the 1st operand of cmpgt
		_mm256_maskstore_epi64((long long *) ptr, _mm256_cmpgt_epi64(vector_t(ptr).vec, vec), vec);
		return *this;
	}
	vector_t &chmax_store(void *ptr) { // faster because cmpgt allows memory address as the 2nd operand
		_mm256_maskstore_epi64((long long *) ptr, _mm256_cmpgt_epi64(vec, vector_t(ptr).vec), vec);
		return *this;
	}
};



// AVX512 / int16_t
template<> class vector_t<InstSet::AVX512, int16_t> : public vector_base_t<InstSet::AVX512> {
public:
	using vector_base_t<InstSet::AVX512>::vector_base_t;
	vector_t (int16_t val) : vector_base_t(_mm512_set1_epi16(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm512_add_epi16(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm512_sub_epi16(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm512_sub_epi16(_mm512_setzero_si512(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm512_min_epi16(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm512_max_epi16(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) {
		_mm512_mask_storeu_epi16((internal_vector_t *) (ptr),
			_mm512_cmp_epi16_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_LT), vec);
		return *this;
	}
	vector_t &chmax_store(void *ptr) {
		_mm512_mask_storeu_epi16((internal_vector_t *) (ptr),
			_mm512_cmp_epi16_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_GT), vec);
		return *this;
	}
};
// AVX512 / int32_t
template<> class vector_t<InstSet::AVX512, int32_t> : public vector_base_t<InstSet::AVX512> {
public:
	using vector_base_t<InstSet::AVX512>::vector_base_t;
	vector_t (int32_t val) : vector_base_t(_mm512_set1_epi32(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm512_add_epi32(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm512_sub_epi32(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm512_sub_epi32(_mm512_setzero_si512(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm512_min_epi32(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm512_max_epi32(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) {
		_mm512_mask_store_epi32((internal_vector_t *) (ptr),
			_mm512_cmp_epi32_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_LT), vec);
		return *this;
	}
	vector_t &chmax_store(void *ptr) {
		_mm512_mask_store_epi32((internal_vector_t *) (ptr),
			_mm512_cmp_epi32_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_GT), vec);
		return *this;
	}
};
// AVX512 / int64_t
template<> class vector_t<InstSet::AVX512, int64_t> : public vector_base_t<InstSet::AVX512> {
public:
	using vector_base_t<InstSet::AVX512>::vector_base_t;
	vector_t (int64_t val) : vector_base_t(_mm512_set1_epi64(val)) {}
	vector_t operator + (const vector_t &rhs) const { return { _mm512_add_epi64(vec, rhs.vec) }; }
	vector_t operator - (const vector_t &rhs) const { return { _mm512_sub_epi64(vec, rhs.vec) }; }
	vector_t operator - () const { return { _mm512_sub_epi64(_mm512_setzero_si512(), vec) }; }
	friend vector_t min(const vector_t &lhs, const vector_t &rhs) { return { _mm512_min_epi64(lhs.vec, rhs.vec) }; }
	friend vector_t max(const vector_t &lhs, const vector_t &rhs) { return { _mm512_max_epi64(lhs.vec, rhs.vec) }; }
	vector_t &chmin_store(void *ptr) {
		_mm512_mask_store_epi64((internal_vector_t *) (ptr),
			_mm512_cmp_epi64_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_LT), vec);
		return *this;
	}
	vector_t &chmax_store(void *ptr) {
		_mm512_mask_store_epi64((internal_vector_t *) (ptr),
			_mm512_cmp_epi64_mask(vec, _mm512_load_si512((internal_vector_t *) ptr), _MM_CMPINT_GT), vec);
		return *this;
	}
};


template<InstSet inst_set, typename T> vector_t<inst_set, T> &operator += (
	vector_t<inst_set, T> &lhs, const vector_t<inst_set, T> &rhs) {
	lhs = lhs + rhs;
	return lhs;
}
template<InstSet inst_set, typename T> vector_t<inst_set, T> &operator -= (
	vector_t<inst_set, T> &lhs, const vector_t<inst_set, T> &rhs) {
	lhs = lhs - rhs;
	return lhs;
}

} // namespace vectorize
} // namespace quick_floyd_warshall
