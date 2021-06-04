#ifndef MIGRAPHX_GUARD_TEST_RUNNER_CMDLINE_OPTIONS_HPP
#define MIGRAPHX_GUARD_TEST_RUNNER_CMDLINE_OPTIONS_HPP
#include <string>

bool cmdOptionExists(char** begin, char** end, const std::string& option);
char* getCmdOption(char** begin, char** end, const std::string& option);

#endif
