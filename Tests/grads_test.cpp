#include <stdlib.h>
#include <stdio.h>
#include <string>

#include "TWeightsInit.h"
#include "Utils.h"

#include "Net.h"
#include "Sequential.h"
#include "Optimizer.h"

using namespace std;

Net createNet(shape input)
{
	Net n;
	n.n_layers = 7;
	n.Layers = (Layer**)malloc(n.n_layers * sizeof(Layer*));
	n.Layers[0] = Input_Create(input);
	n.Layers[1] = Dense_Create(20, n.Layers[0]->out_shape);
	n.Layers[2] = Relu_Create(n.Layers[1]->out_shape);
	n.Layers[3] = Dense_Create(20, n.Layers[2]->out_shape);
	n.Layers[4] = Relu_Create(n.Layers[3]->out_shape);
	n.Layers[5] = Dense_Create(3, n.Layers[4]->out_shape);
	n.Layers[6] = Softmax_Create(n.Layers[5]->out_shape);
	return n;
}

int main()
{
	shape input = { 1,1,100 };
	Net n = createNet(input);
	dList grads = Net_getGradients(&n);
	for (size_t i = 0; i < grads.length; i++)
	{
		Tensor* t = getLElem(Tensor, grads, i);
		TPrint(t);
		float mean = T_Mean(t);
		printf("mean value: %f\n", mean);
	}
	dList_free(&grads);

	//TPrint(&((Dense*)n.Layers[1]->aData)->kernels[0]);
	return 0;
}