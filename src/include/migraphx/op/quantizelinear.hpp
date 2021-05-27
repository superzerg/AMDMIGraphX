#ifndef MIGRAPHX_GUARD_OPERATORS_QUANTIZELINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANTIZELINEAR_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct quantizelinear
{
    int64_t axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "quantizelinear"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        auto type = shape::uint8_type;
        if(inputs.size() == 3)
        {
            type = inputs.at(2).type();
        }

        return {type, inputs.at(0).lens()};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
