#pragma once
#include <random>
#include <chrono>

struct Random {
	std::mt19937 rnd; // intentionally default-constructed for consistent results across executions
	
	int64_t rnd_int(int64_t l, int64_t r) { assert(l <= r); return std::uniform_int_distribution<int64_t>(l, r)(rnd); }
};

struct Timer {
	using clock_type = decltype(std::chrono::high_resolution_clock::now());
	static clock_type get() { return std::chrono::high_resolution_clock::now(); }
	static double diff_s(clock_type start, clock_type end) {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0;
	}
	static double diff_ms(clock_type start, clock_type end) {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.0;
	}
};
