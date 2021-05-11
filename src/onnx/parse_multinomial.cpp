#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_multinomial_op : op_parser<parse_multinomial_op>
{
    std::vector<op_desc> operators() const { return {{"Multinomial", "argmax"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        return info.add_instruction(make_op(opd.op_name, {{"axis", -1}}), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
