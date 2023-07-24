#include "TWeightsInit.h"

float xavier_rand(int n) {
	//calculate the range for the weights
	float lower = -(1.0f / sqrtf((float)n));
	float upper = (1.0f / sqrtf((float)n));
	float num = rngFloat();
	//scale to the desired range
	float scaled = lower + num * (upper - lower);
	return scaled;
}
float xavier_norm_rand(int n, int m)
{
	//calculate the range for the weights
	float lower = -(sqrtf(6.0f) / sqrtf((float)n + (float)m));
	float upper = (sqrtf(6.0f) / sqrtf((float)n + (float)m));
	//get random number
	float num = rngFloat();
	//scale to the desired range
	float scaled = lower + num * (upper - lower);
	return scaled;
}
float he_rand(int n)
{
	//calculate the range for the weights
	float std = sqrtf(2.0f / (float)n);
	//generate random number from a standard normal distribution
	float num = rngNormal();
	//scale to the desired range
	float scaled = num * std;
	return scaled;
}