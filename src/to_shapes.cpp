#include <migraphx/to_shapes.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<shape> to_shapes(const std::vector<argument>& args)
{
    std::vector<shape> vec_s(args.size());
    std::transform(args.begin(), args.end(), vec_s.begin(), [](auto arg) {
        return arg.get_shape();
    });

    return vec_s;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
