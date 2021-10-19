#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/float_equal.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream, const argument& result, const argument& arg_data)
{
    auto s            = arg_data.get_shape();
    auto elem_num     = s.elements();
    auto out_elem_num = result.get_shape().elements();

    int *nonzero_elem_num;
    (void)hipHostMalloc(reinterpret_cast<void **>(&nonzero_elem_num), sizeof(int));
    *nonzero_elem_num = 0;

    // call the prefix_sum function to do a prefix_sum to compute
    // index in the output. Only 1 block can be used since we have
    // only one prefix sum
    const index_int block_size = 256;
    hip_visit_all(arg_data, s)([&](auto input, auto si) {
        const auto* in_ptr = device_cast(input.data());
        auto* ptr          = result.cast<int64_t>();

        gs_launch(stream, elem_num)([=](auto i) {
            if(not float_equal(in_ptr[i], 0))
            {
                atomicAdd(nonzero_elem_num, 1);
            }
        });
        (void)hipStreamSynchronize(stream);

        gs_launch(stream, block_size, block_size)([=](auto, auto idx) __device__ {
            // fill all output to 0 first
            idx.local_stride(out_elem_num, [&](auto j) { ptr[j] = 0; });

            block_scan<block_size>(idx,
                                   sum{},
                                   0,
                                   elem_num,
                                   [&](auto j) { return (float_equal(in_ptr[j], 0)) ? 0 : 1; },
                                   [&](auto j, auto x) {
                                       auto out_loc = x - 1;
                                       if(float_equal(in_ptr[j], 0))
                                           return;

                                       auto index = si.multi(j);
                                       for(size_t k = 0; k < index.size(); ++k)
                                       {
                                           ptr[k * (*nonzero_elem_num) + out_loc] = index[k];
                                       }
                                   });
        });
    });

    const auto& out_shape = result.get_shape();
    auto out_lens = out_shape.lens();
    out_lens[1] = *nonzero_elem_num;
    shape out_s{out_shape.type(), out_lens};
    (void)hipHostFree(nonzero_elem_num);

    return result.reshape(out_s);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
