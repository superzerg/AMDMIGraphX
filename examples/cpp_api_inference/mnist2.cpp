#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>
#include <migraphx/migraphx.hpp>

void compile_model()
{
    migraphx::program prog = migraphx::parse_onnx("mnist-8.onnx");
    migraphx::quantize_fp16(prog);

    migraphx_compile_options comp_opts;
    comp_opts.offload_copy = true;
    prog.compile(migraphx::target("gpu"), comp_opts);

    migraphx::save(prog, "mnist.migraphx");
}

void run_model(std::vector<float> digit)
{
    migraphx::program prog = migraphx::load("mnist.migraphx");
    migraphx::shape input_shape{migraphx_shape_float_type, {1, 1, 28, 28}};
    migraphx::arguments outputs = prog.eval({{"Input3", migraphx::argument{input_shape, digit.data()}}});

    float* results = reinterpret_cast<float*>(outputs[0].data());
    float* max     = std::max_element(results, results + outputs[0].get_shape().bytes() / 4);
    int answer     = max - results;
    std::cout << answer << std::endl;
}

std::vector<float> read_nth_digit(const int n)
{
    std::vector<float> digit;
    const std::string SYMBOLS = "@0#%=+*-.  ";
    std::ifstream file("digits.txt");
    const int DIGITS = 10;
    const int HEIGHT = 28;
    const int WIDTH  = 28;

    if(!file.is_open())
    {
        return digit;
    }

    for(int d = 0; d < DIGITS; ++d)
    {
        for(int i = 0; i < HEIGHT * WIDTH; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            if(d == n)
            {
                float data = temp / 255.0;
                digit.push_back(data);
                std::cout << SYMBOLS[(int)(data * 10) % 11];
                if((i + 1) % WIDTH == 0)
                    std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
    return digit;
}

int main() {
    compile_model();
    run_model(read_nth_digit(1));
}
