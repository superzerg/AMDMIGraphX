#include <migraphx/rewrite_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_qdq::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "quantizelinear")
        {
            apply_quantizelinear(m, ins);
        }
        else if(ins->name() == "dequantizelinear")
        {
            apply_dequantizelinear(m, ins);
        }
    }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void rewrite_qdq::apply_quantizelinear(module& m, instruction_ref ins) const
{
    assert(ins->name() == "quantizelinear");
    auto type = ins->get_shape().type();

    int max_val = 255;
    int min_val = 0;
    if(type == shape::int8_type)
    {
        max_val -= 128;
        min_val -= 128;
    }

    auto inputs  = ins->inputs();
    auto in_lens = inputs[0]->get_shape().lens();
    int n_dim    = static_cast<int>(in_lens.size());

    auto val = ins->get_operator().to_value();
    assert(val.contains("axis"));
    int axis = val.at("axis").to<int>();
    std::vector<std::size_t> dims(in_lens.size(), 1);

    auto scale = inputs[1];
    if(not(scale->get_shape().elements() == 1))
    {
        axis       = tune_axis(n_dim, axis, ins->name());
        dims[axis] = in_lens[axis];
        scale      = m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), scale);
    }

    scale = m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", in_lens}}), scale);
    auto scale_type = scale->get_shape().type();
    if(inputs[0]->get_shape().type() != scale_type)
    {
        inputs[0] =
            m.insert_instruction(ins, make_op("convert", {{"target_type", scale_type}}), inputs[0]);
    }
    auto div            = m.insert_instruction(ins, make_op("div"), inputs[0], scale);
    auto div_round      = m.insert_instruction(ins, make_op("round"), div);
    auto add_zero_point = div_round;
    auto s              = add_zero_point->get_shape();

    if(inputs.size() == 3)
    {
        auto zero_point = inputs[2];
        if(not(zero_point->get_shape().elements() == 1))
        {
            axis       = tune_axis(n_dim, axis, ins->name());
            dims[axis] = in_lens[axis];
            zero_point =
                m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), zero_point);
        }
        zero_point = m.insert_instruction(
            ins, make_op("multibroadcast", {{"output_lens", in_lens}}), zero_point);
        if(scale_type != zero_point->get_shape().type())
        {
            zero_point = m.insert_instruction(
                ins, make_op("convert", {{"target_type", scale_type}}), zero_point);
        }
        add_zero_point = m.insert_instruction(ins, make_op("add"), add_zero_point, zero_point);
    }

    const auto& lens = s.lens();
    std::vector<int64_t> out_lens(lens.begin(), lens.end());
    std::vector<int> min_data(s.elements(), min_val);
    std::vector<int> max_data(s.elements(), max_val);
    auto min_arg = m.add_literal(literal(s, min_data));
    auto max_arg = m.add_literal(literal(s, max_data));

    auto saturated = m.insert_instruction(ins, make_op("clip"), add_zero_point, min_arg, max_arg);
    m.replace_instruction(ins, make_op("convert", {{"target_type", type}}), saturated);
}

void rewrite_qdq::apply_dequantizelinear(module& m, instruction_ref ins) const
{
    assert(ins->name() == "dequantizelinear");
    auto inputs  = ins->inputs();
    auto in_lens = inputs[0]->get_shape().lens();
    int n_dim    = static_cast<int>(in_lens.size());
    auto type    = ins->get_shape().type();

    auto val = ins->get_operator().to_value();
    assert(val.contains("axis"));
    int axis = val.at("axis").to<int>();
    std::vector<std::size_t> dims(in_lens.size(), 1);

    auto sub_zero_point = inputs[0];
    if(sub_zero_point->get_shape().type() != type)
    {
        sub_zero_point =
            m.insert_instruction(ins, make_op("convert", {{"target_type", type}}), sub_zero_point);
    }
    if(inputs.size() == 3)
    {
        auto zero_point = inputs[2];
        if(not(zero_point->get_shape().elements() == 1))
        {
            axis       = tune_axis(n_dim, axis, ins->name());
            dims[axis] = in_lens[axis];

            zero_point =
                m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), zero_point);
        }
        zero_point = m.insert_instruction(
            ins, make_op("multibroadcast", {{"output_lens", in_lens}}), zero_point);
        if(zero_point->get_shape().type() != type)
        {
            zero_point =
                m.insert_instruction(ins, make_op("convert", {{"target_type", type}}), zero_point);
        }
        sub_zero_point = m.insert_instruction(ins, make_op("sub"), sub_zero_point, zero_point);
    }

    auto scale = inputs[1];
    if(not(scale->get_shape().elements() == 1))
    {
        axis       = tune_axis(n_dim, axis, ins->name());
        dims[axis] = in_lens[axis];
        scale      = m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), scale);
    }
    scale = m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", in_lens}}), scale);
    if(scale->get_shape().type() != type)
    {
        scale = m.insert_instruction(ins, make_op("convert", {{"target_type", type}}), scale);
    }

    m.replace_instruction(ins, make_op("mul"), sub_zero_point, scale);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
