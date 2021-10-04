#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/roctx_mark.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    roctx_mark marker;
    marker.initalize_roctx();
    marker.mark("Marker demo: marked for softmax.");
    uint64_t range_id = marker.range_start("Marker demo: range started");

    
    auto batch_lens          = result.get_shape().lens();
    index_int batch_item_num = batch_lens[axis];
    batch_lens[axis]         = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {
        const index_int max_block_size = 120;

        
        marker.trace_ins_start("Marker start: Softmax: Compute_block_size()");
        const index_int block_size     = compute_block_size(batch_item_num, max_block_size);
        marker.trace_ins_end();

        gs_launch(stream,
                  batch_shape.elements() * block_size,
                  block_size)([=](auto i, auto idx) __device__ __host__ {
            auto data_idx = batch.multi(i / block_size);
            using type    = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
            type init     = lowest();

            //marker.trace_ins_start("Marker start: Softmax: Compute_block_size()  ");
            auto batch_max = block_reduce<max_block_size>(
                idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                    data_idx[axis] = j;
                    return input[data_idx];
                });
            //marker.trace_ins_end();

            //marker.trace_ins_start("Marker start: Softmax: Compute_block_size()    ");
            auto batch_sum =
                block_reduce<max_block_size>(idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                    data_idx[axis] = j;
                    auto val       = input[data_idx] - batch_max;
                    return ::exp(to_hip_type(val));
                });
            //marker.trace_ins_end();
            
            //marker.trace_ins_start("Marker start: Softmax: Compute_block  _size() ");
            idx.local_stride(batch_item_num, [&](auto j) __device__ {
                data_idx[axis]   = j;
                auto val         = input[data_idx] - batch_max;
                output[data_idx] = ::exp(to_hip_type(val)) / batch_sum;
            });
            //marker.trace_ins_end();

        });
    });
    marker.range_stop(range_id);

}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
