#include "migraphx/gpu/hip.hpp"
#include <migraphx/run_loop.hpp>
#include <migraphx/gpu/loop.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/fill.hpp>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_loop::compute_shape(std::vector<shape> inputs, std::vector<module_ref> mods) const
{
    auto input_num = (inputs.size() - 2) / 2;
    inputs.erase(inputs.begin() + input_num, inputs.end());
    return op.compute_shape(inputs, std::move(mods));
}

struct gpu_loop
{
    int64_t max_iterations = 0;

    template <class T>
    void copy(context& ctx, const argument& src, T& dst) const
    {
        argument arg_dst{src.get_shape(), &dst};
        copy_from_gpu(ctx, src, arg_dst);
    }

    template <class T>
    void copy(context& ctx, T src, const argument& dst) const
    {
        argument arg_src{dst.get_shape(), &src};
        copy_to_gpu(ctx, arg_src, dst);
    }

    void append(context& ctx,
                const std::vector<argument>& iter_state,
                const std::vector<argument>& concatenated_outputs,
                int iter,
                const std::vector<int>& indices) const
    {
        for(int j = 0; j < concatenated_outputs.size(); ++j)
        {
            std::cout << "concat_data_ptr = " << (void*)concatenated_outputs[j].data() << std::endl;
            std::cout << "concat_scan_out_" << j << " = "
                      << migraphx::gpu::from_gpu(concatenated_outputs[j]) << std::endl;
        }

        for(auto i : range(iter_state.size()))
        {
            if(contains(indices, i))
                continue;
            const auto& iter_stat = iter_state.at(i);
            const auto& scan_out  = concatenated_outputs.at(i);

            auto* in_data        = iter_stat.data();
            auto* out_data       = scan_out.data();
            std::size_t out_size = iter_stat.get_shape().bytes();
            assert((iter + 1) * out_size <= scan_out.get_shape().bytes());
            (void)hipMemcpyAsync(out_data + iter * out_size,
                                 in_data,
                                 out_size,
                                 hipMemcpyDeviceToDevice,
                                 ctx.get_stream().get());
        }
    }

    void set_zero(context& ctx, const std::vector<argument>& concatenated_outputs, int iter) const
    {
        if(iter >= max_iterations)
            return;

        auto elem_num = max_iterations - iter;
        for(const auto& out : concatenated_outputs)
        {
            auto s    = out.get_shape();
            auto size = s.bytes() / max_iterations;
            auto lens = s.lens();
            lens[0]   = elem_num;
            shape ss{s.type(), lens};
            assert(ss.bytes() + iter * size <= out.get_shape().bytes());
            device::fill(ctx.get_stream().get(), argument(ss, out.data() + iter * size), 0);
        }
    }

    std::unordered_map<std::string, int> get_output_params(const module& m) const
    {
        auto get_output_index = [](const std::string& name) {
            std::string out_prefix = "#output_";
            auto loc               = name.find(out_prefix);
            if(loc != std::string::npos)
            {
                int index = std::stoi(name.substr(loc + out_prefix.size()));
                return index;
            }

            return -1;
        };

        const auto& param_names = m.get_parameter_names();
        std::unordered_map<std::string, int> result;
        for(const auto& name : param_names)
        {
            auto index = get_output_index(name);
            if(index == -1)
                continue;
            result[name] = index;
        }

        return result;
    }
};

argument
hip_loop::compute(context& ctx,
                  const shape&,
                  const std::vector<argument>& args,
                  const std::vector<module_ref>& mods,
                  const std::function<std::vector<argument>(
                      module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
{
    auto out_args = run_loop(gpu_loop{op.max_iterations}, ctx, args, mods, run);

    for(int i = 0; i < args.size() - 1; ++i)
    {
        std::cout << "loop_args_" << i << " = " << migraphx::gpu::from_gpu(args[i]) << std::endl;
    }

    auto&& oargs = out_args.get_sub_objects();
    for(int i = 0; i < oargs.size(); ++i)
    {
        std::cout << "loop_out_" << i << " = " << migraphx::gpu::from_gpu(oargs[i]) << std::endl;
    }

    return {out_args};

    // return run_loop(gpu_loop{op.max_iterations}, ctx, args, mods, run);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
