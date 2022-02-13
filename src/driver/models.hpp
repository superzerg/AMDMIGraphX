
#include <migraphx/program.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

migraphx::program resnet50(int batch);
migraphx::program inceptionv3(int batch);
migraphx::program alexnet(int batch);

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
