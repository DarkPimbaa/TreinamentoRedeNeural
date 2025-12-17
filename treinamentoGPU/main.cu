#define NUM_ENTRADAS 60 // 6 valores por candle * 100 candles
#define NUM_CAMADAS 4
#define NUM_SAIDAS 2
#define NUM_INDIVIDUOS 1000 //
#define META_TAXA_VITORIA 60

#include <cstdio>
#include "includes/redeNeural.cu"
#include "includes/types.hpp"
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include "./includes/utils.hpp"

__global__ void initCurand(curandState *estados, unsigned long seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &estados[tid]);
}

// TODO fazer todos os kernels necessarios

int main() {
  size_t sizeCandle = (sizeof(Candle) * 964800);
  size_t sizeIndividuos = (sizeof(RedeNeural) * NUM_INDIVIDUOS);

  RedeNeural* h_individuos = new RedeNeural[NUM_INDIVIDUOS];
  Candle* h_candles = new Candle[964800];

  lerCSV_mallocc("./filtrado.csv", h_candles, 964800);

  // initCurand
  curandState *d_estados;
  cudaMalloc(&d_estados,NUM_INDIVIDUOS * sizeof(curandState)); // 1 por indivíduo
  initCurand<<<NUM_INDIVIDUOS, 1>>>(d_estados, time(NULL));
  cudaDeviceSynchronize();

  // TODO implementar a lógica de treino

  delete[] h_candles;
  delete[] h_individuos;
  cudaFree(d_estados);

  return 0;
}