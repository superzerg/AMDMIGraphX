#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_SCHEDULE_MODEL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_SCHEDULE_MODEL_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct operation;

namespace gpu {

struct schedule_model
{
    int streams = 0;
    int concurrency() const;
    void sched(module& p, instruction_ref ins, int n) const;
    void wait(module& p, instruction_ref ins, int wait_id) const;
    void record(module& p, instruction_ref ins, int wait_id) const;
    int weight(const operation& op) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
