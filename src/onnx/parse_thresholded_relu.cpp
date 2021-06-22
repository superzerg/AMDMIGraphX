#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_thresholded_relu : op_parser<parse_thresholded_relu>
{
    std::vector<op_desc> operators() const { return {{"ThresholdedRelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        float alpha = 1.0f;
        if(contains(info.attributes, "alpha"))
        {
            alpha = info.attributes.at("alpha").f();
        }

        auto s = args.front()->get_shape();
        std::vector<float> vec(s.elements(), alpha);
        auto la = info.add_literal(literal(s, vec));
        auto x  = info.add_instruction(make_op("sub"), args.front(), la);
        return info.add_instruction(make_op("relu"), x);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
