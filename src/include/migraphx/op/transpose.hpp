#ifndef MIGRAPHX_GUARD_OPERATORS_TRANSPOSE_HPP
#define MIGRAPHX_GUARD_OPERATORS_TRANSPOSE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/lifetime.hpp>
#include <migraphx/to_shapes.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct transpose
{
    std::vector<int64_t> dims;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dims, "permutation"));
    }

    std::string name() const { return "transpose"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input         = inputs.at(0);
        auto input_lens    = input.lens();
        auto input_strides = input.strides();
        auto t             = input.type();

        if(dims.size() != input_lens.size())
        {
            MIGRAPHX_THROW("Permutation has wrong number of axes");
        }
        std::vector<int64_t> axes(dims.size());
        std::iota(axes.begin(), axes.end(), 0);
        if(!std::is_permutation(axes.begin(), axes.end(), dims.begin()))
        {
            MIGRAPHX_THROW("TRANSPOSE: Invalid permutation");
        }
        std::vector<size_t> output_lens(input_lens.size());
        std::vector<size_t> output_strides(input_lens.size());
        for(std::size_t i = 0; i < output_lens.size(); i++)
        {
            output_lens[i]    = input_lens[dims[i]];
            output_strides[i] = input_strides[dims[i]];
        }
        return {t, output_lens, output_strides};
    }
    argument compute(shape, std::vector<argument> args) const
    {
        auto in_ss = to_shapes(args);
        auto out_s = compute_shape(in_ss);
        return args[0].reshape(out_s);
    }
    lifetime get_lifetime() const { return lifetime::borrow; }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
