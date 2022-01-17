#include <migraphx/gpu/concat.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/concat.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_concat::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.normalize_compute_shape(inputs);
}

argument hip_concat::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    auto vec_ss = to_shapes(args);
    vec_ss.pop_back();
    auto output_shape                = op.normalize_compute_shape(vec_ss);
    std::vector<std::size_t> offsets = op.compute_offsets(output_shape, args);
    return device::concat(ctx.get_stream().get(), output_shape, args, offsets);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
