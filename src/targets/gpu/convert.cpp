#include <migraphx/gpu/convert.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/convert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_convert::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    check_shapes{inputs, *this}.packed();
    return op.compute_shape(inputs);
}

argument hip_convert::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto result = args.at(1).share();
    device::convert(ctx.get_stream().get(), result, args[0]);
    return result;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
