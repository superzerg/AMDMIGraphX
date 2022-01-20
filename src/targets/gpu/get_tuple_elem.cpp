#include <migraphx/gpu/get_tuple_elem.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/contiguous.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_get_tuple_elem::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument
hip_get_tuple_elem::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto sub_args = args.front().get_sub_objects();
    device::contiguous(ctx.get_stream().get(), args.back(), sub_args[op.index]);
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
