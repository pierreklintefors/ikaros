#include "ikaros.h"

using namespace ikaros;

class AveragePool : public Module
{
    parameter size;
    parameter stride;

    matrix input;
    matrix output;
    matrix sums; // summed-area table of one channel, one row and column larger than the input

    int windowSize = 2;
    int windowStride = 2;


    // Writes the mean of each window of source into destination; both are rank 2. The summed-area
    // table gives each window's sum from four values, so the cost does not grow with the window size.
    void
    Pool(const matrix & source, matrix & destination)
    {
        const int rows = source.rows();
        const int cols = source.cols();
        const int stride = cols + 1;
        float * table = sums.data();
        const float * values = source.is_contiguous() ? source.data() : nullptr;
        for(int x = 0; x <= cols; x++)
            table[x] = 0.0f;
        for(int y = 0; y < rows; y++)
        {
            const float * above = table + y * stride;
            float * row = table + (y + 1) * stride;
            float rowSum = 0.0f;
            row[0] = 0.0f;
            for(int x = 0; x < cols; x++)
            {
                rowSum += values ? values[y * cols + x] : source(y, x);
                row[x + 1] = above[x + 1] + rowSum;
            }
        }

        const float scale = 1.0f / static_cast<float>(windowSize * windowSize);
        for(int r = 0; r < destination.rows(); r++)
        {
            const int top = r * windowStride;
            const int bottom = top + windowSize;
            for(int c = 0; c < destination.cols(); c++)
            {
                const int left = c * windowStride;
                const int right = left + windowSize;
                destination(r, c) = scale * (table[bottom * stride + right] - table[top * stride + right] -
                                             table[bottom * stride + left] + table[top * stride + left]);
            }
        }
    }

public:
    void
    Init() override
    {
        Bind(size, "size");
        Bind(stride, "stride");
        Bind(input, "INPUT");
        Bind(output, "OUTPUT");

        windowSize = size.as_int();
        windowStride = stride.as_int();
        if(windowSize < 1 || windowStride < 1)
            throw exception("AveragePool: size and stride must be at least 1.", path_);
        if(input.rank() != 2 && input.rank() != 3)
            throw exception("AveragePool: INPUT must be [rows, cols] or [channels, rows, cols].", path_);
        if(input.rows() < windowSize || input.cols() < windowSize)
            throw exception("AveragePool: INPUT is smaller than one window.", path_);

        const int rows = (input.rows() - windowSize) / windowStride + 1;
        const int cols = (input.cols() - windowSize) / windowStride + 1;
        const bool shapeOK = input.rank() == 2
            ? output.rank() == 2 && output.rows() == rows && output.cols() == cols
            : output.rank() == 3 && output.size_z() == input.size_z() && output.rows() == rows && output.cols() == cols;
        if(!shapeOK)
            throw exception("AveragePool: OUTPUT does not have the expected shape.", path_);

        sums.realloc(input.rows() + 1, input.cols() + 1);
    }


    void
    Tick() override
    {
        if(input.rank() == 2)
            Pool(input, output);
        else
            for(int i = 0; i < input.size_z(); i++)
            {
                matrix source = input[i];
                matrix destination = output[i];
                Pool(source, destination);
            }
    }
};

INSTALL_CLASS(AveragePool)
