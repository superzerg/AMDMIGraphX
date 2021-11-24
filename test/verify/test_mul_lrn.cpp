
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

struct test_mul_lrn : verify_program<test_mul_lrn>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {1, 5, 2, 2}};
        auto x = mm->add_parameter("x", s);
        std::vector<float> vec(s.elements(), 500.0f);
        auto l  = mm->add_literal(migraphx::literal(s, vec));
        auto xl = mm->add_instruction(migraphx::make_op("mul"), x, l);
        auto y  = mm->add_instruction(migraphx::make_op("relu"), xl);
        mm->add_instruction(
            migraphx::make_op("lrn",
                              {{"alpha", 0.0001}, {"beta", 0.75}, {"bias", 1.0}, {"size", 5}}),
            y);
        return p;
    }
};
