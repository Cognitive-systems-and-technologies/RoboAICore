#include "Model.h"

void WeightsInitTest()
{
	int count = 10000;
	float min = FLT_MAX, max = FLT_MIN, sum = 0;
	printf("xavier\n");
	for (size_t i = 0; i < count; i++)
	{
		float f = xavier_rand(10);
		//printf("%f, ", f);
		min = f < min ? f : min;
		max = f > max ? f : max;
		sum += f;
	}
	printf("\nmin val:%f max val:%f, mean:%f\n", min, max, sum / (float)count);
	min = FLT_MAX, max = FLT_MIN, sum = 0;
	printf("\nxavier normalized\n");
	for (size_t i = 0; i < count; i++)
	{
		float f = xavier_norm_rand(10, 20);
		//printf("%f, ", f);
		min = f < min ? f : min;
		max = f > max ? f : max;
		sum += f;
	}
	printf("\nmin val:%f max val:%f, mean:%f\n", min, max, sum / (float)count);
	min = FLT_MAX, max = FLT_MIN, sum = 0;
	printf("\nhe distribution\n");
	for (size_t i = 0; i < count; i++)
	{
		float f = he_rand(10);
		//printf("%f, ", f);
		min = f < min ? f : min;
		max = f > max ? f : max;
		sum += f;
	}
	printf("\nmin val:%f max val:%f, mean:%f\n", min, max, sum / (float)count);
}

int main()
{
	WeightsInitTest();

	printf("\nPress enter to close...");
	getchar();
	return 0;
}