// optimization techniques are taken from the following files 
// part1_reduce_double.cpp
// part2_reduce_double.cpp
// random number generator: rng.cpp

#include "stdio.h"
#include <HLS/hls.h>
#include <HLS/hls_float.h>
#include <HLS/hls_float_math.h>
#include "HLS/rand_lib.h"

// this was in the part1/2_native_type.cpp examples but I couldnt find it in the reference value
#define _USE_MATH_DEFINES // need to define this for Windows
#include <math.h>

// rounding mode : rounds to the nearest zero value. this is not as accurate as RNE (nearest, tie to even), but uses less FPGA are which was prioritized here 
constexpr auto Rnd = ihc::fp_config::FP_Round::RZERO;
// IEEE 754 half-precision (binary 16)
using half_float  = ihc::hls_float<5, 10, Rnd>;

// seed value for random number generator 
#define SEED                1543

half_float component dfr() {
	 const int initSamples = 200,
				trainingSamples = 400,
				// total samples
				numSamples = initSamples + trainingSamples, 
				// number of nodes
				N = 40;
	
	// mg activation function constants
	half_float gamma = 0.05, 
			   eta = 0.5;
			   
	// these variables are not used 
	// half_float tau = 80, 			// delay
	// 		   theta = tau/N;			// degrees of seperation

	half_float u[numSamples], 			// narma10 input vector 
               y[numSamples],			// narma10 output vector (expected output)
               W[N],				    // weights
               M[N],					// mask
               J[numSamples][N],		// J = M * u
               X[N],					// temporary vector that helps us achieve feedback (400th node in this case) 
               XHistory[trainingSamples][N];

    // random number generator seed 
    // not sure if using half_float here will work 
    // static does not restart the series with every invocation
    // automatically generates between [0, 1)
    static RNG_Uniform<float> rng(SEED);
    // this is used only to generate random weights for W
    static RNG_Uniform<int> rngInt(SEED);

	// creating the narma10 input and expected output vectors
    // input
	for (int i = 0; i < numSamples; i++)
		u[i] = rng.rand() * 0.5 ;

    // narma 10 output
	for (int i = 9; i < numSamples - 1; i++) {
		half_float sum = 0;
        // sum of last 10 terms 
		for (int j = 0; j < 9; j++)
			sum += y[i - j];
		y[i + 1] = (0.3 * y[i]) + (0.05 * y[i] * sum) + (1.5 * u[i] * u[i - 9]) + 0.1;
	}

	half_float rndMultiplier = 2/N;
	// generating mask values and weights
	for (int i = 0; i < N; i++) {
		M[i] = (rngInt.rand() % 2 == 0) ? -0.1 : 0.1;
		W[i] = rng.rand() * ihc::ihc_sqrt(rndMultiplier);
	}

	// generating masked input matrix
	for (int i = 0; i < numSamples; i++) {
		for (int j = 0; j < N; j++) {
			J[i][j] = M[j] * u[i];
		}
	}

	// reservoir initialization and training
	for (int i = 0; i < numSamples; i++) {
		for (int j = 0; j < N; j++) {
			// evaluating current input 
			// half_float temp = mg(gamma * J[i][j] + eta * X[N - 1]);
			half_float x = (gamma * J[i][j] + eta * X[N - 1]);
			half_float temp = (1.33 * x) / (1 + 0.4 * x);
			// shifting the values in time 
			for (int k = N-1; k > 0; k--)
				X[k] = X[k - 1];
			// adding new value into the network
			X[0] = temp;
		}
		// dont have to add the values into the XHistory vector during initialization (typo)
		if (i >= initSamples) {
			for (int j = 0; j < N; j++)
				XHistory[i - initSamples][j] = X[j];
        }
	}

	// weight training using gradient descent
	// vector of only the training samples 
    half_float yTrain[trainingSamples];
	for (int i = 0; i < trainingSamples; i++) {
		yTrain[i] = y[i + initSamples];
	}
	
	int iterations = 10; 
	int learning_rate = 50;

	half_float y_hat_reg[trainingSamples];
	// gradient descent iterations
	for (int i = 0; i < iterations; i++) {
		// calculate preditcted outputs 
		for (int j = 0; j < trainingSamples; j++) {
			half_float dot = 0;
            // dot product
			for (int k = 0; k < N; k++)
				dot += W[k] * XHistory[j][k];
			y_hat_reg[j] = dot;
		}

		// calculate the weight updates
		for (int i = 0; i < N; i++) {
			// calculating the mean 
			half_float XHistMean = 0;
            // dot product
			for (int j = 0; j < trainingSamples; j++)
				XHistMean += (y_hat_reg[i] - yTrain[i]) * XHistory[j][i];
			XHistMean /= trainingSamples;

			W[i] = W[i] - learning_rate * XHistMean;
		}
	}

	// calculate the NRMSE values (forbenius norm)
	half_float numerator = 0, denominator = 0;
	for(int i = 0; i < trainingSamples; i++) {
		numerator += ihc::ihc_pown(y_hat_reg[i] - yTrain[i], 2);
		denominator += ihc::ihc_pown(yTrain[i], 2);
	}
	numerator = ihc::ihc_sqrt(numerator);
	denominator = ihc::ihc_sqrt(denominator);

	return numerator / denominator;
}

int main() {
    double nrmse = dfr();

    printf("NRMSE: %f", nrmse);

    return EXIT_SUCCESS;
}