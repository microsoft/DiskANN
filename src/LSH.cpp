#include <memory>
#include <cstdint>
#include <random>
#include <mkl.h>

#include "utils.h"
#include "lsh.h"

namespace diskann
{

void generate_random_vectors(uint32_t dim, uint32_t num, float *vecs, float min_value = FLT_TRUE_MIN,
                             float max_value = FLT_MAX, bool normalized = false)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dis(min_value, max_value);

    for (uint32_t i = 0; i < num; i++)
    {
        for (uint32_t j = 0; j < dim; j++)
        {
            vecs[i * dim + j] = dis(rng);
        }
        if (normalized)
        {
            for (uint32_t i = 0; i < num; i++)
                normalize(vecs + (dim * i), dim);
        }
    }
}

LSH::LSH(uint32_t num_dims, uint32_t num_axes) : _num_axes(num_axes), _num_dims(num_dims)
{
    _axes = std::make_unique<float[]>(num_axes * num_dims);
    generate_random_vectors(num_dims, num_axes, const_cast<float *>(_axes.get()), -1, 1, false);
}

LSH::LSH(uint32_t num_dims, uint32_t num_axes, float *axes) : _num_axes(num_axes), _num_dims(num_dims)
{
    _axes.reset(axes);
}

uint8_t LSH::get_hash(const float *vec, float *dot_product)
{
    memset(dot_product, 0, _num_axes * sizeof(float));

    // computing Y = alpha * A (mat) * X (vec) + beta * Y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, _num_axes /* M */, _num_dims /* N */, 1.0f, /* alpha */
                const_cast<float*>(_axes.get()),                                         /* A */
                _num_dims,                                                               /* lda */
                vec,                                                                     /* X */
                1,                                                                       /* inc x */
                0,                                                                       /* Beta */
                dot_product,                                                             /* Y */
                1                                                                        /* inc Y */
    );
    uint8_t lsh_value = 0;
    for (uint32_t i = 0; i < _num_axes; i++)
    {
        lsh_value = dot_product[i] < 0 ? lsh_value : lsh_value | (1 << i);
    }
    return lsh_value;
}

void LSH::dump_axes_to_text_file(std::ostream &out_stream)
{
    for (uint32_t i = 0; i < _num_axes; i++)
    {
        for (uint32_t j = 0; j < _num_dims; j++)
        {
            out_stream << _axes[i * _num_dims + j] << " ";
        }
        out_stream << std::endl;
    }
}
} //namespace diskann

