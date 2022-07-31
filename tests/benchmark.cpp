#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "quick_floyd_warshall/qfw.h"
#include "utils/utils.h"

using namespace quick_floyd_warshall;

template<class Runner> unsigned int benchmark(Random &random, int n, bool symmetric, double *time) {
	using value_t = typename Runner::value_t;
	std::vector<value_t> mat(n * n);
	for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) mat[i * n + j] = mat[j * n + i] = random.rnd_int(1, Runner::INF / n);
	for (int i = 0; i < n; i++) mat[i * n + i] = 0;
	
	auto start = Timer::get();
	for (int i = 0; i < 1; i++) Runner::run(n, mat.data(), mat.data(), symmetric);
	auto end = Timer::get();
	
	if (time) *time = Timer::diff_s(start, end);
	
	uint32_t hash = mat.size();
	for (auto i : mat) hash ^= std::hash<unsigned int>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
	return hash;
}

template<class Runner> bool test_multi(int n, int n_testcase, bool symmetric = false) {
	Random random;
	
	std::vector<double> time(n_testcase);
	for (int i = 0; i < n_testcase; i++) {
		unsigned int hash = benchmark<Runner>(random, n, symmetric, &time[i]);
		fprintf(stderr, "%u ", hash);
	}
	fprintf(stderr, "\n");
	
	std::sort(time.begin(), time.end());
	for (auto &i : time) i *= 1000; // convert to ms
	double mean = std::accumulate(time.begin(), time.end(), 0.0) / n_testcase;
	double sd = 0;
	for (auto i : time) sd += (i - mean) * (i - mean);
	sd /= n_testcase;
	sd = sqrt(sd);
	
	printf("[%7.2f ms | %7.2f ms | %7.2f ms]  Avg:%7.2f ms  SD:%.2f ms\n", time.front(), time[time.size() / 2], time.back(), mean, sd);
	// for (auto i : time) std::cerr << "  " << (i - time.front()) * 1000 << " us" << std::endl;
	
	return true;
}

int main() {
	int size, n_testcase;
	scanf("%d%d", &size, &n_testcase);
	
	test_multi<floyd_warshall_naive<int64_t> >(size, n_testcase);
	test_multi<floyd_warshall<InstSet::AVX2, int64_t, 3> >(size, n_testcase);
	return 0;
}
