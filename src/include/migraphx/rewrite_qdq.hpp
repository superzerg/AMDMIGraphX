#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_QDQ_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_QDQ_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite Q/DQ operators to the combination of other operators
 */
struct rewrite_qdq
{
    std::string name() const { return "rewrite_qdq"; }
    void apply(module& m) const;

    private:
    // for quantizelinear and dequantizelinear operator
    void apply_quantizelinear(module& m, instruction_ref ins) const;
    void apply_dequantizelinear(module& m, instruction_ref ins) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
