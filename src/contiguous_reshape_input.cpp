#include <migraphx/contiguous_reshape_input.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void contiguous_reshape_input::apply(module& p) const
{
    for(auto ins : reverse_iterator_for(p))
    {
        if(ins->name() != "reshape")
            continue;

        auto input = ins->inputs().front();
        if(input->name() == "contiguous")
            continue;

        auto cinput = p.insert_instruction(std::next(input), make_op("contiguous"), input);
        p.replace_instruction(input, cinput);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
