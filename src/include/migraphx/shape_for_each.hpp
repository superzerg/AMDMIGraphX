#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_SHAPE_FOR_EACH_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_SHAPE_FOR_EACH_HPP

#include <migraphx/shape.hpp>
#include <migraphx/config.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F>
void shape_for_each(const migraphx::shape& s, F f)
{
    // Ensure calls to f use const ref to vector
    auto call = [&f](const std::vector<int>& i) { f(i); };
    std::vector<int> indices(s.lens().size());
    shape ss{s.type(), s.lens()};
    for(int i = 0; i < ss.elements(); i++)
    {
        std::transform(ss.strides().begin(),
                       ss.strides().end(),
                       ss.lens().begin(),
                       indices.begin(),
                       [&](int stride, int len) {
                           assert(len > 0 and stride > 0);
                           return (i / stride) % len;
                       });
        call(indices);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
