#include <cstdio>
#define NUM_ENTRADAS 60
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2
#define BIAS 1.f
#define NUM_INDIVIDUOS 1000

#include <cstddef>
#include "includes/redeNeural.cu"
#include "includes/types.hpp"
#include "includes/utils.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

__global__ void treinamento(Candle* d_candles, RedeNeural* d_individuos, int n){
    int individuo = blockIdx.x;     // 0–999  || warps
    int lane      = threadIdx.x;    // 0–31   || threads

    // não deixa passar do limite de warps
    if (individuo >= n) return;

    // só o thread 0 faz o trampo para o warp agir como unidade.
    if (lane == 0) {
        
    }
};

int main(){
    size_t sizeCandle = (sizeof(Candle) * 964800);
    size_t sizeIndividuos = (sizeof(RedeNeural) * NUM_INDIVIDUOS);
    Candle* h_candles = (Candle*)malloc(sizeCandle);
    RedeNeural* h_individuos = new RedeNeural[NUM_INDIVIDUOS];
    lerCSV_mallocc("./filtrado.csv", h_candles, 964800);

    // manda os candles para memoria da gpu
    Candle* d_candles;
    cudaMalloc(&d_candles, sizeCandle);
    cudaMemcpy(d_candles, h_candles, sizeCandle, cudaMemcpyHostToDevice);

    // manda os individuos já com pesos inicializados para a memoria da gpu
    RedeNeural* d_individuos;
    cudaMalloc(&d_individuos, sizeIndividuos);
    cudaMemcpy(d_individuos, h_individuos, sizeIndividuos, cudaMemcpyHostToDevice);

    //dim3 bloco(32);
    //dim3 grid(1000);

    //treinamento<<<grid, bloco>>>(d_candles, d_individuos, NUM_INDIVIDUOS);
    //cudaDeviceSynchronize();

    std::cout << (sizeIndividuos / (1024 * 1024));

    free(h_candles);
    delete[] h_individuos;
    cudaFree(d_candles);
    cudaFree(d_individuos);

    return 0;
}