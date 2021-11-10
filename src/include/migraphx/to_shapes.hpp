#ifndef MIGRAPHX_GUARD_RTGLIB_TO_SHAPES_HPP
#define MIGRAPHX_GUARD_RTGLIB_TO_SHAPES_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::vector<shape> to_shapes(const std::vector<argument>& args);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
