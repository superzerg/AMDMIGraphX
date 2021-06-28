
#include <migraphx/ref/target.hpp>
#include <migraphx/ref/lowering.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/rewrite_rnn.hpp>
<<<<<<< HEAD
#include <migraphx/rewrite_qdq.hpp>
=======
#include <migraphx/eliminate_pad.hpp>
#include <migraphx/insert_pad.hpp>
>>>>>>> c44112297f591e350ea35d39fe4b97be2af14a33
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/normalize_ops.hpp>

namespace migraphx
{
    inline namespace MIGRAPHX_INLINE_NS {
    namespace ref {

    std::string target::name() const { return "ref"; }

    std::vector<pass> target::get_passes(migraphx::context&, const compile_options&) const
    {
        return {normalize_ops{},
                rewrite_qdq{},
                dead_code_elimination{},
                eliminate_pad{},
                dead_code_elimination{},
                insert_pad{},
                dead_code_elimination{},
                rewrite_rnn{},
                dead_code_elimination{},
                auto_contiguous{},
                dead_code_elimination{},
                lowering{},
                dead_code_elimination{}};
    }

    argument target::allocate(const shape& s) const { return fill_argument(s, 0); }

    MIGRAPHX_REGISTER_TARGET(target);

    } // namespace ref
    } // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
