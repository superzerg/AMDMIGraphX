#include "get_cases.hpp"
#include <algorithm>
#include <iostream>
#include <experimental/filesystem>

namespace fs = ::std::experimental::filesystem;

std::string get_path_last_folder(const std::string& path_str)
{
    const fs::path p = path_str;
    std::string last = *std::prev(p.end());

    return last;
}

std::vector<std::string> get_test_cases(const std::string& path_str)
{
    std::vector<std::string> sub_dirs;
    for (const auto& entry : fs::directory_iterator(path_str))
    {
        const auto& entry_path = entry.path();
        bool is_dir = fs::is_directory(entry_path);
        if (is_dir)
        {
            sub_dirs.push_back(entry_path);
        }
    }
    std::sort(sub_dirs.begin(), sub_dirs.end());
    return sub_dirs;
}

std::string get_model_name(const std::string& path_str)
{
    for (const auto& entry : fs::directory_iterator(path_str))
    {
        const auto& entry_path = entry.path();
        bool is_file = fs::is_regular_file(entry_path);
        if (is_file)
        {
            auto ext = fs::path(entry_path).extension();
            if (ext == ".onnx")
            {
                return entry_path;
            }
        }
    }

    return {};
}


