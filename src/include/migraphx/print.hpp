#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_PRINT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_PRINT_HPP

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <ostream>

template <class T>
void print_vec(std::ostream& os, const std::vector<T>& vec)
{
    std::size_t print_num = 320;
    os << "{";
    std::size_t elem_num = vec.size() > print_num ? print_num : vec.size();
    for(std::size_t i = 0; i < elem_num; ++i)
    {
        os << std::setw(12) << vec[i];
        if(i != vec.size() - 1)
            os << ", ";
        if(((i + 1) % 8) == 0)
            os << std::endl;
    }

    std::cout << "........." << std::endl;

    std::size_t start = vec.size() - print_num > 0 ? vec.size() - print_num : 0;
    start             = start < print_num ? print_num : start;
    for(std::size_t i = 0; i < elem_num; ++i)
    {
        os << std::setw(12) << vec[i];
        if(i != vec.size() - 1)
            os << ", ";
        if(((i + 1) % 8) == 0)
            os << std::endl;
    }
    os << "}";
}

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
    print_vec(os, vec);
    return os;
}

#endif
