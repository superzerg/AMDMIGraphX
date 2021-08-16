
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_conv3 : verify_program<test_conv3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto input =
            mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 256, 1, 1}});
        auto weights =
            mm->add_parameter("w", migraphx::shape{migraphx::shape::float_type, {256, 256, 3, 3}});
        mm->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", {2, 2, 2, 2}}, {"stride", {1, 1}}, {"dilation", {2, 2}}}),
            input,
            weights);
        return p;
    }
};
