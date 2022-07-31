#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "quick_floyd_warshall/qfw.h"
#include "utils/utils.h"

using namespace quick_floyd_warshall;

typedef enum {
	RANDOM_DENSE,
	RANDOM_PATH,
	MAX_PATH
} GraphType;
template<class CorrectRunner> struct Test {
	using value_t = typename CorrectRunner::value_t;
	static constexpr value_t INF = CorrectRunner::INF;
	
	int n;
	bool symmetric;
	std::vector<value_t> org_matrix;
	std::vector<value_t> correct_matrix;
	std::vector<value_t> test_matrix;
	Test (Random &random, int n_low, int n_high, bool symmetric, GraphType graph_type) {
		n = random.rnd_int(n_low, n_high);
		this->symmetric = symmetric;
		value_t MAX_UNIFORM_WEIGHT = (INF - 1) / std::max(1, n - 1);
		if (graph_type == GraphType::RANDOM_DENSE) {
			org_matrix.assign(n * n, 0);
			if (symmetric) {
				for (int i = 0; i < n; i++) for (int j = 0; j < i; j++)
					org_matrix[i * n + j] = org_matrix[j * n + i] = random.rnd_int(1, MAX_UNIFORM_WEIGHT);
			} else {
				for (int i = 0; i < n; i++) for (int j = 0; j < n; j++)
					org_matrix[i * n + j] = random.rnd_int(1, MAX_UNIFORM_WEIGHT);
			}
		} else if (graph_type == GraphType::RANDOM_PATH || graph_type == GraphType::MAX_PATH) {
			org_matrix.assign(n * n, (value_t) INF); // avoid referencing a constexpr variable
			std::vector<int> perm(n);
			std::iota(perm.begin(), perm.end(), 0);
			for (int i = 1; i < n; i++) std::swap(perm[random.rnd_int(0, i)], perm[i]);
			for (int i = 0; i + 1 < n; i++) {
				org_matrix[perm[i] * n + perm[i + 1]] = graph_type == GraphType::RANDOM_PATH ?
					random.rnd_int(1, MAX_UNIFORM_WEIGHT) : MAX_UNIFORM_WEIGHT;
				if (symmetric) org_matrix[perm[i + 1] * n + perm[i]] = org_matrix[perm[i] * n + perm[i + 1]];
			}
		}
		
		correct_matrix = org_matrix;
		CorrectRunner::run(n, correct_matrix.data(), correct_matrix.data(), symmetric);
	}
	template<class TestRunner> bool test() {
		static_assert(std::is_same<typename CorrectRunner::value_t, typename TestRunner::value_t>::value, "value_t mismatch");
		test_matrix = org_matrix;
		TestRunner::run(n, test_matrix.data(), test_matrix.data(), symmetric);
		if (test_matrix != correct_matrix) {
			int diff_cnt = 0;
			for (int i = 0; i < n * n; i++) diff_cnt += test_matrix[i] != correct_matrix[i];
			printf("\n%s FAILED: %d elements differ\n", TestRunner::get_description().c_str(), diff_cnt);
			for (int i = 0; i < n * n; i++) if (test_matrix[i] != correct_matrix[i]) {
				printf("  first mismatch at #%d((%d, %d)) : %lld (correct: %lld)\n", i, i / n, i % n,
					(long long) test_matrix[i], (long long) correct_matrix[i]);
				break;
			}
			return false;
		} else return true;
	}
};

template<InstSet inst_set, typename test_t> bool test_all_unroll_types(test_t &test) {
	if (!test.template test<floyd_warshall<inst_set, typename test_t::value_t, 0> >()) return false;
	if (!test.template test<floyd_warshall<inst_set, typename test_t::value_t, 1> >()) return false;
	if (!test.template test<floyd_warshall<inst_set, typename test_t::value_t, 2> >()) return false;
	if (!test.template test<floyd_warshall<inst_set, typename test_t::value_t, 3> >()) return false;
	return true;
}
template<typename test_t> bool test_all_instruction_sets(test_t &test) {
	if (!test_all_unroll_types<InstSet::DEFAULT, test_t>(test)) return false;
	if (!test_all_unroll_types<InstSet::SSE4_2 , test_t>(test)) return false;
	if (!test_all_unroll_types<InstSet::AVX2   , test_t>(test)) return false;
	return true;
}

template<typename T> bool test_all_multiple(Random &random, int n_low, int n_high, int n_tests, bool symmetric, GraphType graph_type) {
	constexpr int DOT_OMIT_THREASHOLD = 30;
	
	printf("  Test n:[%d, %d] x%d ", n_low, n_high, n_tests);
	
	bool res = true;
	int pass_cnt = 0;
	int last_displayed_pass_cnt_len = 0;
	const int pass_cnt_interval = std::max(1, n_tests / 100);
	for (int t = 0; t < n_tests; t++) {
		Test<floyd_warshall_naive<T> > test(random, n_low, n_high, symmetric, graph_type);
		if (!test_all_instruction_sets<>(test)) {
			res = false;
			break;
		}
		
		pass_cnt++;
		if (pass_cnt < DOT_OMIT_THREASHOLD) printf(".");
		else if (pass_cnt % pass_cnt_interval == 0) {
			std::string output_str = std::string(last_displayed_pass_cnt_len, '\b') + "x" + std::to_string(pass_cnt);
			printf("%s", output_str.c_str());
			last_displayed_pass_cnt_len = ("x" + std::to_string(pass_cnt)).size();
		}
	}
	if (res) puts(" OK");
	return res;
}
template<typename T> bool test_all_with_standard_parameters(Random &random, bool symmetric, GraphType graph_type) {
	printf("Testing int%d_t...\n", (int) (sizeof(T) * 8));
	return 
		test_all_multiple<T>(random, 500, 600, 1 , symmetric, graph_type) &&
		test_all_multiple<T>(random, 200, 500, 4 , symmetric, graph_type) &&
		test_all_multiple<T>(random, 100, 200, 10, symmetric, graph_type) &&
		test_all_multiple<T>(random, 32, 100, 100, symmetric, graph_type) &&
		test_all_multiple<T>(random, 1, 32, 1000 , symmetric, graph_type);
}

int main() {
	Random random;
	for (int symmetric = 0; symmetric < 2; symmetric++) {
		if (!test_all_with_standard_parameters<int64_t>(random, symmetric, GraphType::RANDOM_DENSE)) return 1;
		if (!test_all_with_standard_parameters<int32_t>(random, symmetric, GraphType::RANDOM_DENSE)) return 1;
		if (!test_all_with_standard_parameters<int16_t>(random, symmetric, GraphType::RANDOM_DENSE)) return 1;
		
		if (!test_all_with_standard_parameters<int64_t>(random, symmetric, GraphType::RANDOM_PATH)) return 1;
		if (!test_all_with_standard_parameters<int32_t>(random, symmetric, GraphType::RANDOM_PATH)) return 1;
		if (!test_all_with_standard_parameters<int16_t>(random, symmetric, GraphType::RANDOM_PATH)) return 1;
		
		if (!test_all_with_standard_parameters<int64_t>(random, symmetric, GraphType::MAX_PATH)) return 1;
		if (!test_all_with_standard_parameters<int32_t>(random, symmetric, GraphType::MAX_PATH)) return 1;
		if (!test_all_with_standard_parameters<int16_t>(random, symmetric, GraphType::MAX_PATH)) return 1;
	}
	return 0;
}
