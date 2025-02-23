#include <migraphx/gpu/argmax.hpp>
#include <migraphx/gpu/device/argmax.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_argmax::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(2);
    return op.normalize_compute_shape({inputs.at(0)});
}

argument hip_argmax::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto n_dim         = args.front().get_shape().lens().size();
    int64_t tuned_axis = tune_axis(n_dim, op.axis, op.name());
    device::argmax(ctx.get_stream().get(), args.back(), args.front(), tuned_axis);
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
