#ifndef MIGRAPHX_GUARD_OPERATORS_SHAPE_OP_HPP
#define MIGRAPHX_GUARD_OPERATORS_SHAPE_OP_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/context.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct shape_op
{
    std::string name() const { return "shape"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        std::vector<std::size_t> lens = {inputs[0].lens().size()};
        return {shape::int64_type, lens};
    }

    argument compute(context&, const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto lens = args.front().get_shape().lens();

        result.visit([&](auto v) { std::copy(lens.begin(), lens.end(), v.begin()); });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
