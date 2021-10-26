#include <migraphx/gpu/device/convert.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void convert(hipStream_t stream, argument& result, const argument& arg)
{
    result.visit([&](auto output) {
        arg.visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr      = device_cast(output.data());
            gs_launch(stream, arg.get_shape().elements())(
                [=](auto i) __device__ { output_ptr[i] = input_ptr[i]; });
        });
    });

    if(result.get_shape() != arg.get_shape())
    {
        auto lens = arg.get_shape().lens();
        shape out_s{result.get_shape().type(), lens};
        result = result.reshape(out_s);
    }
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
