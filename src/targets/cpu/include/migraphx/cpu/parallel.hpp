#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP

// #define MIGRAPHX_DISABLE_OMP

#include <migraphx/config.hpp>
#ifdef MIGRAPHX_DISABLE_OMP
#include <migraphx/par_for.hpp>
#else
#include <omp.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

#ifdef MIGRAPHX_DISABLE_OMP

inline int max_threads() { return std::thread::hardware_concurrency(); }

template <class F>
void parallel_for_impl(int n, int threadsize, F f)
{
    if(threadsize <= 1)
    {
        f(int{0}, n);
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
// Using const here causes gcc 5 to ICE
#if(!defined(__GNUC__) || __GNUC__ != 5)
        const
#endif
            int grainsize = std::ceil(static_cast<double>(n) / threads.size());

        int work = 0;
        std::generate(threads.begin(), threads.end(), [=, &work] {
            auto result =
                joinable_thread([=]() mutable { f(work, std::min(n, work + grainsize)); });
            work += grainsize;
            return result;
        });
        // cppcheck-suppress unsignedLessThanZero
        assert(work >= n);
    }
}
#else

inline int max_threads() { return omp_get_max_threads(); }

template <class F>
void parallel_for_impl(int n, int threadsize, F f)
{
    if(threadsize <= 1)
    {
        f(int{0}, n);
    }
    else
    {
        int grainsize = std::ceil(static_cast<double>(n) / threadsize);
#pragma omp parallel for num_threads(threadsize) schedule(static, 1) private(grainsize, n)
        for(int tid = 0; tid < threadsize; tid++)
        {
            int work = tid * grainsize;
            f(work, std::min(n, work + grainsize));
        }
    }
}
#endif
template <class F>
void parallel_for(int n, int min_grain, F f)
{
    const auto threadsize = std::min<int>(max_threads(), n / min_grain);
    parallel_for_impl(n, threadsize, f);
}

template <class F>
void parallel_for(int n, F f)
{
    const int min_grain = 8;
    parallel_for(n, min_grain, f);
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
