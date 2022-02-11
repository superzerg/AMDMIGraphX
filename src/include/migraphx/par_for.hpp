#ifndef MIGRAPHX_GUARD_RTGLIB_PAR_FOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_PAR_FOR_HPP

#include <thread>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) // NOLINT
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <class F>
auto thread_invoke(int i, int tid, F f) -> decltype(f(i, tid))
{
    f(i, tid);
}

template <class F>
auto thread_invoke(int i, int, F f) -> decltype(f(i))
{
    f(i);
}

template <class F>
void par_for_impl(int n, int threadsize, F f)
{
    if(threadsize <= 1)
    {
        for(int i = 0; i < n; i++)
            thread_invoke(i, 0, f);
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
        int tid  = 0;
        std::generate(threads.begin(), threads.end(), [=, &work, &tid] {
            auto result = joinable_thread([=] {
                int start = work;
                int last  = std::min(n, work + grainsize);
                for(int i = start; i < last; i++)
                {
                    thread_invoke(i, tid, f);
                }
            });
            work += grainsize;
            ++tid;
            return result;
        });
        assert(work >= n);
    }
}

template <class F>
void par_for(int n, int min_grain, F f)
{
    const auto threadsize = std::min<int>(std::thread::hardware_concurrency(),
                                                  n / std::max<int>(1, min_grain));
    par_for_impl(n, threadsize, f);
}

template <class F>
void par_for(int n, F f)
{
    const int min_grain = 8;
    par_for(n, min_grain, f);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
