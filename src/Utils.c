#include "Utils.h"
#include <stdlib.h>
#include <stdio.h>

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