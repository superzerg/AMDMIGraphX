#include <migraphx/simplify_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

std::unordered_set<std::string> get_all_op_names()
{
    static auto ops = get_operators();
    static std::unordered_set<std::string> s(ops.begin(), ops.end());
    return s;
}

bool inputs_are_zeros(const instruction_ref& ins)
{
    literal zp_lit;
    if(ins->name() == "multibroadcast" or ins->name() == "broadcast")
        zp_lit = ins->inputs().at(0)->get_literal();
    else
        zp_lit = ins->get_literal();
    std::vector<int64_t> zero_points;
    zp_lit.visit([&](const auto zp) {
        std::transform(
            zp.begin(), zp.end(), std::back_inserter(zero_points), [](auto&& z) { return z; });
    });
    return std::all_of(
        zero_points.begin(), zero_points.end(), [](const auto& z) { return z == 0; });
}

double get_scale(const instruction_ref& ins)
{
    literal scale_lit;
    if(ins->name() == "multibroadcast" or ins->name() == "broadcast")
        scale_lit = ins->inputs().at(0)->get_literal();
    else
        scale_lit = ins->get_literal();
    std::vector<float> scales;
    scale_lit.visit([&](auto sl) {
        std::transform(
            sl.begin(), sl.end(), std::back_inserter(scales), [](auto&& s) { return s; });
    });
    double epsilon = 1e-6;
    if(not std::all_of(scales.begin(), scales.end(), [&](const auto& s) {
           return std::abs(s - scales.front()) < epsilon;
       }))
        MIGRAPHX_THROW("Multiple scales not currently supported");
    return scales.front();
}

instruction_ref insert_quantize_op(module& m,
                                   instruction_ref ins,
                                   const std::string& name,
                                   instruction_ref x,
                                   instruction_ref scale,
                                   instruction_ref shift)
{
    auto lens = x->get_shape().lens();
    auto scale_mb =
        m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), scale);
    auto shift_mb =
        m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), shift);
    return m.insert_instruction(ins, make_op(name), x, scale_mb, shift_mb);
}

struct match_quantizable_ops
{
    auto matcher() const
    {
        match::name(get_quantizable_op_names())(
            match::arg(0)(match::name("dequantizelinear")(
                match::arg(0)(match::name("quantizelinear")).bind("q0"))).bind("dq0"),
            match::arg(1)(match::name("dequantizelinear")(
                match::arg(0)(match::name("quantizelinear")).bind("q1"))).bind("dq1"));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto ins  = r.result;
        auto dq0 = r.instructions["dq0"];
        auto q0  = r.instructions["q0"];
        auto dq1 = r.instructions["dq1"];
        auto q1  = r.instructions["q1"];

        // Only INT8 type currently supported
        if(q0->get_shape().type() != migraphx::shape::int8_type or
           q1->get_shape().type() != migraphx::shape::int8_type)
            return;

        std::vector<instruction_ref> lits = {q0, dq0, q1, dq1};
        if(not std::all_of(lits.begin(), lits.end(), [&](auto in) {
            const auto& lit_inputs = in->inputs();
            return (lit_inputs.size() == 2) or inputs_are_zeros(lit_inputs.at(2));
        }))
        {
            return;
        }

        // Only zero_point==0 currently supported
        auto dq0_args  = dq0->inputs();
        auto dq1_args  = dq1->inputs();

        auto scale     = get_scale(dq0_args.at(1)) * get_scale(dq1_args.at(1));
        auto qop_args  = ins->inputs();
        qop_args.at(0) = q0;
        qop_args.at(1) = q1;
        auto lens = ins->get_shape().lens();

        instruction_ref qop{};
        instruction_ref dq_scale{};
        instruction_ref zero_point{};
        if(ins->name() == "convolution")
        {
            auto conv_val = ins->get_operator().to_value();
            qop = m.insert_instruction(ins, make_op("quant_convolution", conv_val), qop_args);
            dq_scale     = m.add_literal(static_cast<float>(scale));
            zero_point   = m.add_literal(0);
        }
        else if(qop->name() == "dot")
        {
            auto dot_op    = any_cast<op::dot>(qop->get_operator());
            auto scale_val = dot_op.alpha / scale;
            instruction_ref input_c = m.end();
            if (qop_args.size() == 3)
            {
                input_c = qop_args.at(2);
                qop_args.pop_back();
            }

            qop = m.insert_instruction(ins, make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qop_args);
            dq_scale   = m.add_literal(static_cast<float>(scale_val));
            if (dot_op.beta != 0.0f and input_c != m.end())
            {
                auto l_beta = m.add_literal(-1.0f * dot_op.beta / scale_val);
                auto m_beta = m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), l_beta);
                zero_point = m.insert_instruction(ins, make_op("mul"), m_beta, input_c);
            }
            else
            {
                zero_point = m.add_literal(0.0f);
            }
        }

        auto scale_mb = m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), dq_scale);
        if (zero_point->get_shape().lens() != lens)
        {
            zero_point = m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), zero_point);
        }

        m.replace_instruction(ins, make_op("dequantizelinear"), qop, scale_mb, zero_point);
    }
};

struct remove_qdq_pair
{
    auto matcher() const
    {
            match::name("dequantizelinear")(match::arg(0)(match::name("quantizelinear")));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto dq  = r.result;
        auto inputs = dq->inputs();
        auto q = inputs.at(0);
        auto input = q->inputs().at(0);
        m.replace_instruction(dq, input);
    }
};

void simplify_qdq::apply(module& m) const
{
    match::find_matches(m, match_quantizable_ops{}, remove_qdq_pair{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
