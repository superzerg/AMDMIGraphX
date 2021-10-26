#include <migraphx/gpu/device/sigmoid.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void sigmoid(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x)
                                  __device__ { 
        if(x >= 0)
        {
            return 1.f / (1.f + ::exp(to_hip_type(-x)));
        }
        else
        {
            auto v = exp(to_hip_type(x));
            return v / (1.0f + v);            
        }
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
