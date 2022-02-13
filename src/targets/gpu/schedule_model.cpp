#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct record_event
{
    int event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::record_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.get_stream().record(ctx.get_event(event));
        return {};
    }

    void finalize(context& ctx, const shape&, const std::vector<shape>&) const
    {
        ctx.create_events(event);
    }
};

struct wait_event
{
    int event = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.event, "event"));
    }
    std::string name() const { return "gpu::wait_event"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.get_stream().wait(ctx.get_event(event));
        return {};
    }
};

struct set_stream
{
    int stream = 0;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.stream, "stream"));
    }
    std::string name() const { return "gpu::set_stream"; }
    shape compute_shape(const std::vector<shape>&) const { return {}; }

    argument compute(context& ctx, const shape&, const std::vector<argument>&) const
    {
        ctx.set_stream(stream);
        return {};
    }
    void finalize(context& ctx, const shape&, const std::vector<shape>&) const
    {
        ctx.set_stream(stream);
    }
};

MIGRAPHX_REGISTER_OP(record_event)
MIGRAPHX_REGISTER_OP(wait_event)
MIGRAPHX_REGISTER_OP(set_stream)

int schedule_model::concurrency() const { return streams; }
void schedule_model::sched(module& p, instruction_ref ins, int n) const
{
    auto last_stream = std::find_if(std::make_reverse_iterator(ins),
                                    std::make_reverse_iterator(p.begin()),
                                    [&](auto&& i) { return i.name() == "gpu::set_stream"; });
    if(last_stream != std::make_reverse_iterator(p.begin()))
    {
        auto&& op = any_cast<set_stream>(last_stream->get_operator());
        // If the same stream was set earlier then skip
        if(op.stream == n)
            return;
    }
    p.insert_instruction(ins, set_stream{n});
}

void schedule_model::wait(module& p, instruction_ref ins, int wait_id) const
{
    p.insert_instruction(ins, wait_event{wait_id});
}
void schedule_model::record(module& p, instruction_ref ins, int wait_id) const
{
    p.insert_instruction(std::next(ins), record_event{wait_id});
}

static std::unordered_map<std::string, int> create_weight_map()
{
    return {{"hip::load_literal", 0},
            {"hip::hip_allocate_memory", 0},
            {"hip::hip_load_memory", 0},
            {"hip::allocate", 0},
            {"gpu::convolution", 8},
            {"gpu::conv_bias_relu", 8},
            {"gpu::pooling", 4},
            {"gpu::gemm", 4}};
}

static const std::unordered_map<std::string, int>& weight_map()
{
    static const std::unordered_map<std::string, int> m = create_weight_map();
    return m;
}

int schedule_model::weight(const operation& op) const
{
    if(weight_map().count(op.name()) == 0)
    {
        return 2;
    }
    return weight_map().at(op.name());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
