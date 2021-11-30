#ifndef MIGRAPHX_GUARD_RTGLIB_CONTIGUOUS_RESHAPE_INPUT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTIGUOUS_RESHAPE_INPUT_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct contiguous_reshape_input
{
    std::string name() const { return "contiguous_reshape_input"; }
    void apply(module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
