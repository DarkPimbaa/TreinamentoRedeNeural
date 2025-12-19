#define NUM_ENTRADAS 60 // 6 valores por candle * 100 candles
#define NUM_CAMADAS 4
#define NUM_SAIDAS 2
#define NUM_INDIVIDUOS 1'000 //
#define META_TAXA_VITORIA 60

#include <cstdio>
#include "includes/redeNeural.cu"
#include "includes/types.hpp"
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include "./includes/utils.hpp"
#include <chrono>
#include <iostream>

__global__ void initCurand(curandState *estados, unsigned long seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &estados[tid]);
}

// cada thread é uma rede
__global__ void iniciarPesos(RedeNeural *d_individuos, int tamanho, curandState *state){
  int idx = threadIdx.x;
  if (idx < tamanho) {
    d_individuos[idx].iniciarPesosDevice(d_individuos[idx].rede, state);
  }
}

// cada thread cuida de um indice de valor
__global__ void knormalizarValores(Candle *d_candles, int indiceAtual, float *d_valoresNormalizados) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_ENTRADAS) return;

    int candleOffset = tid / 6;   // 0..9
    int campo        = tid % 6;   // 0..5

    Candle c = d_candles[indiceAtual - candleOffset];
    float factor = d_candles[indiceAtual].fechamento;

    switch (campo) {
        case 0: d_valoresNormalizados[tid] = c.abertura   / factor; break;
        case 1: d_valoresNormalizados[tid] = c.maxima     / factor; break;
        case 2: d_valoresNormalizados[tid] = c.minima     / factor; break;
        case 3: d_valoresNormalizados[tid] = c.fechamento / factor; break;
        case 4: d_valoresNormalizados[tid] = c.volume;              break;
        case 5: d_valoresNormalizados[tid] = c.trades;              break;
    }
}

// cada thread é uma rede
__global__ void kalimentarEntrada(RedeNeural *d_individuos, float *d_valoresNormalizados, int tamanho){
  int idx = threadIdx.x;
  for (size_t i = 0; i < NUM_ENTRADAS; i++) {
    d_individuos[idx].rede.entrada[i] = d_valoresNormalizados[i];
  }
}

// cada bloco é uma rede, cada thread um neuronio.
__global__ void kprimeiraInferencia(RedeNeural *d_individuos){
  int rede = blockIdx.x;
  d_individuos[rede].inferenciaPrimeiraOculta();
}

// cada bloco é uma rede, cada thread um neuronio
__global__ void kinferenciaOculta(RedeNeural *d_individuos){
  int rede = blockIdx.x;
  for (size_t camada = 1; camada < NUM_CAMADAS; camada++) {
    d_individuos[rede].inferenciaOculta(camada);
  }
}

// cada thread é uma rede
__global__ void kinferenciaSaida(RedeNeural *d_individuos, Candle *d_candle, int indiceAtual){
  int idx = threadIdx.x;
  
  d_individuos[idx].inferenciaSaida();

  if (d_individuos[idx].bvivo == true) {
  
    if (d_individuos[idx].retorno[0] == true && d_individuos[idx].retorno[1] == false) {
      //compra
      if (d_candle[indiceAtual].fechamento < d_candle[indiceAtual + 1].fechamento) {
        d_individuos[idx].ganho++;
        d_individuos[idx].rodadasSemApostar = 0;
      }else {
        d_individuos[idx].perda++;
        d_individuos[idx].rodadasSemApostar = 0;
      }

    }else if(d_individuos[idx].retorno[0] == false && d_individuos[idx].retorno[1] == true){
      //venda
      if (d_candle[indiceAtual].fechamento > d_candle[indiceAtual + 1].fechamento) {
        d_individuos[idx].ganho++;
        d_individuos[idx].rodadasSemApostar = 0;
      }else {
        d_individuos[idx].perda++;
        d_individuos[idx].rodadasSemApostar = 0;
      }
    }else{
      //nada
      if ( d_individuos[idx].rodadasSemApostar == 200) {
        d_individuos[idx].bvivo = false;
      }
      d_individuos[idx].rodadasSemApostar++;
    }
  }

}

// cada bloco é uma rede, cada thread um neuronio
__global__ void kmutarPesosOCultos(RedeNeural *d_individuos, float por, curandState *state){
  int rede = blockIdx.x;
  for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
    d_individuos[rede].mutacaoDeviceOculta(por, state, camada);
  }
}

// cada thread é uma rede
__global__ void kmutarBias(RedeNeural *d_individuos, float por, curandState *state){
  int idx = threadIdx.x;
  d_individuos[idx].mutarBias(por, state);
}

// cada thread é uma rede
__global__ void kmutarSaida(RedeNeural *d_individuos, float por, curandState *state){
  int idx = threadIdx.x;
  d_individuos[idx].mutarSaida(por, state);
}

// cada thread é uma rede
__global__ void kresetPontuacao(RedeNeural *d_individuos){
  int idx= threadIdx.x;
  d_individuos[idx].bvivo = true;
  d_individuos[idx].ganho = 0;
  d_individuos[idx].perda = 0;
  d_individuos[idx].rodadasSemApostar = 0;
  d_individuos[idx].taxaDeVitoria = 0.f;
}


int main() {
  float taxaDoMelhorDaGeracaoAtual = 0;
  int ganhos = 0;
  int perdas = 0;
  size_t sizeCandle = (sizeof(Candle) * 964800);
  size_t sizeIndividuos = (sizeof(RedeNeural) * NUM_INDIVIDUOS);

  RedeNeural* h_individuos = new RedeNeural[NUM_INDIVIDUOS];
  Candle* h_candles = new Candle[964800];
  RedeNeural* h_melhorIndividuo = new RedeNeural();
  h_melhorIndividuo->taxaDeVitoria = -1.0f;

  lerCSV_mallocc("./filtrado.csv", h_candles, 964800);

  float* d_valoresNormalizados;
  cudaMalloc(&d_valoresNormalizados, sizeof(float) * NUM_ENTRADAS);

  RedeNeural* d_individuos;
  cudaMalloc(&d_individuos, sizeIndividuos);
  Candle* d_candles;
  cudaMalloc(&d_candles, sizeCandle);
  cudaMemcpy(d_candles, h_candles, sizeCandle, cudaMemcpyHostToDevice);
  cudaMemcpy(d_individuos, h_individuos, sizeIndividuos, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // initCurand
  curandState *d_estados;
  cudaMalloc(&d_estados,NUM_INDIVIDUOS * sizeof(curandState)); // 1 por indivíduo
  initCurand<<<NUM_INDIVIDUOS, 1>>>(d_estados, time(NULL));
  cudaDeviceSynchronize();
  iniciarPesos<<<1, NUM_INDIVIDUOS>>>(d_individuos, NUM_INDIVIDUOS, d_estados);
  cudaDeviceSynchronize();

  bool bmetaAtingida = false;
  int geracao = 0;
  // loop de treinamento
  while (!bmetaAtingida) {
    // loop de inferencia
    for (size_t rodada = NUM_ENTRADAS / 6; rodada < 10'000; rodada++) {
      knormalizarValores<<<1, NUM_ENTRADAS>>>(d_candles, rodada, d_valoresNormalizados);
      cudaDeviceSynchronize();
      kalimentarEntrada<<<1, NUM_INDIVIDUOS>>>(d_individuos, d_valoresNormalizados, NUM_INDIVIDUOS);
      cudaDeviceSynchronize();
      kprimeiraInferencia<<<NUM_INDIVIDUOS, NUM_ENTRADAS>>>(d_individuos);
      cudaDeviceSynchronize();
      kinferenciaOculta<<<NUM_INDIVIDUOS, NUM_ENTRADAS>>>(d_individuos);
      cudaDeviceSynchronize();
      kinferenciaSaida<<<1,NUM_INDIVIDUOS>>>(d_individuos, d_candles, rodada);
      cudaDeviceSynchronize();
      
    }

    cudaMemcpy(h_individuos, d_individuos, sizeIndividuos, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // verifica quem é o melhor individuo
    for (size_t rede = 0; rede < NUM_INDIVIDUOS; rede++) {

      for (size_t redee = 0; redee < NUM_INDIVIDUOS; redee++) {
        h_individuos[redee].taxaDeVitoria = ((h_individuos[redee].ganho / (float)(h_individuos[redee].ganho + h_individuos[redee].perda)) * 100);
      }

      taxaDoMelhorDaGeracaoAtual = 0;
      ganhos = 0;
      perdas = 0;
      for (size_t melhor = 0; melhor < NUM_INDIVIDUOS; melhor++) {
        if (h_individuos[melhor].bvivo == true && h_individuos[melhor].taxaDeVitoria > taxaDoMelhorDaGeracaoAtual) {
          taxaDoMelhorDaGeracaoAtual = h_individuos[melhor].taxaDeVitoria;
          ganhos = h_individuos[melhor].ganho;
          perdas = h_individuos[melhor].perda;
        }
      }
      
      if (h_individuos[rede].taxaDeVitoria > h_melhorIndividuo->taxaDeVitoria) {
        
        if (h_individuos[rede].bvivo == true) {
        
          *h_melhorIndividuo = h_individuos[rede];
        }
        
      }
    }

    // replica o melhor individuo
    for (size_t rede = 0; rede < NUM_INDIVIDUOS; rede++) {
      h_individuos[rede] = *h_melhorIndividuo;
    }

    cudaMemcpy(d_individuos, h_individuos, sizeIndividuos, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    //aplicar mutações;
    float por = Rand::Float(0.1, 0.5);
    kmutarPesosOCultos<<<NUM_INDIVIDUOS,NUM_ENTRADAS>>>(d_individuos, por, d_estados);
    cudaDeviceSynchronize();
    kmutarBias<<<1,NUM_INDIVIDUOS>>>(d_individuos, por, d_estados);
    cudaDeviceSynchronize();
    kmutarSaida<<<1,NUM_INDIVIDUOS>>>(d_individuos, por, d_estados);
    cudaDeviceSynchronize();
    kresetPontuacao<<<1, NUM_INDIVIDUOS>>>(d_individuos);
    cudaDeviceSynchronize();


    if (h_melhorIndividuo->taxaDeVitoria >=META_TAXA_VITORIA) {
      printf("META ATINGIDA: %f \n", h_melhorIndividuo->taxaDeVitoria);
      bmetaAtingida = true;
    }


    printf("geração: %i \n", geracao);
    printf("taxa do melhor da geracao atual: %f \n", taxaDoMelhorDaGeracaoAtual);
    printf("ganho do melhor da geracao atual: %i \n", ganhos);
    printf("perda do melhor da geracao atual: %i \n", perdas);
    printf("taxa de vitória do melhor individuo: %f \n", h_melhorIndividuo->taxaDeVitoria);
    printf("---------------------------------------------------------- \n");

    geracao++;
  }

  delete[] h_candles;
  delete[] h_individuos;
  delete h_melhorIndividuo;
  cudaFree(d_valoresNormalizados);
  cudaFree(d_estados);
  cudaFree(d_individuos);
  cudaFree(d_candles);

  return 0;
}