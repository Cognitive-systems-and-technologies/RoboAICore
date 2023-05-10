#include "Utils.h"
#include <stdlib.h>
#include <stdio.h>

void TPrint(Tensor* x)
{
    printf("[");
    for (size_t d = 0; d < x->s.d; d++)
    {
        printf("[");
        for (size_t i = 0; i < x->s.w; i++)
        {
            for (size_t j = 0; j < x->s.h; j++)
            {
                printf("%f,", Tensor_Get(x, i, j, d));
            }
            //printf("\n");
        }
        printf("]");
    }
    printf("]\n");
}

void shuffle(TPair* array, int n)
{
    if (n > 1)
    {
        int i;
        for (i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            TPair t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}