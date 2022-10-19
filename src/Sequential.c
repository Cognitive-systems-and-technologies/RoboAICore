#include "Sequential.h"
#include <stdlib.h>

Tensor* Seq_Forward(Net* n, Tensor* x, int is_training)
{
    Tensor* y = x;
    //forward
    for (size_t i = 0; i < n->n_layers; i++)
    {
        y = Forward_Layer(n->Layers[i], y);
    }
    return y;
}

float Seq_Backward(Net* n, Tensor* y)
{
    int N = n->n_layers;
    float loss = Backward_Layer(n->Layers[N - 1], y); // last layer assumed to be loss layer
    for (int i = N - 2; i >= 0; i--)
    {
        Backward_Layer(n->Layers[i], y);
    }
    return loss;
}

