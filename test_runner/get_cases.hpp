#ifndef MIGRAPHX_GUARD_TEST_RUNNER_GET_CASES_HPP
#define MIGRAPHX_GUARD_TEST_RUNNER_GET_CASES_HPP

#include <string>
#include <vector>

std::vector<std::string> get_test_cases(const std::string& path_str);
std::string get_path_last_folder(const std::string& path_str);
std::string get_model_name(const std::string& path_str);

#endif
