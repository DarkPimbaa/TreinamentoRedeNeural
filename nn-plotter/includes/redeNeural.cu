#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef NUM_ENTRADAS
#define NUM_ENTRADAS 64 // multiplos de 32, 64 neuronios + 1 bias;
#endif
#ifndef NUM_CAMADAS
#define NUM_CAMADAS 4
#endif
#ifndef NUM_SAIDAS
#define NUM_SAIDAS 2
#endif
#ifndef BIAS
#define BIAS 1.f
#endif


struct Rede{
  __half pesos[NUM_CAMADAS][NUM_ENTRADAS][NUM_ENTRADAS];
  __half bias[NUM_CAMADAS][NUM_ENTRADAS];
  __half saida[NUM_SAIDAS][NUM_ENTRADAS];
};

enum Decisao{
  COMPROU, // comprou
  VENDEU, // vendeu
  NADA // não jogou na rodada anterior
};

struct RedeNeural{
  Rede rede;
  uint16_t ganho; // de 0 até 65k
  uint16_t perda; // de 0 até 65k
  uint16_t partidaSemJogar;
  float taxaVitoria;
  bool bvivo;
  uint8_t resultado[NUM_SAIDAS]; // resultado da inferencia anterior

  __device__ __host__ void init(){
    ganho = 0;
    perda = 0;
    partidaSemJogar = 0;
    bvivo = true;
    taxaVitoria = 0;
  }

  __device__ float attTaxa(){
    int total = ganho + perda;
    if (total == 0) {
    taxaVitoria = 0.0f;
    return taxaVitoria;
    }
    taxaVitoria = ((float)ganho / total) * 100;
    return taxaVitoria;
  }

  /**
   * @brief inicia os pesos de toda a rede, ideal usar um kernel thread/rede
   * 
   * @param states curandStates*
   */
  __device__ void iniciarPesos(curandState *states){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    init();

    // gera os pesos de pesos[][][]
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
        for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
          rede.pesos[camada][neuronio][peso] = ((curand_uniform(&states[i]) * 2.f) - 1.f);
        }
      }
    }

    // gera os pesos de bias[][]
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.bias[camada][peso] = ((curand_uniform(&states[i]) * 2.f) - 1.f);
      }
    }

    // gera os pesos de saida[][]
    for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.saida[neuronio][peso] = ((curand_uniform(&states[i]) * 2.f) - 1.f);
      }
    }
    //__syncthreads();
  }

  /**
   * @brief faz a inferencia na rede ideal block/rede
   * 
   * @param valores array com os valores normalizados
   * @param tamanhoDeValores tamanho do array valores
   * @return bool* retorna um array de boleanos
   */
  __device__ void inferencia(__half *valores, uint8_t camada){
    int idx = threadIdx.x;

    if (idx >= NUM_ENTRADAS) return;

    __half valor = 0.0;
    for (uint16_t i = 0; i < NUM_ENTRADAS; i++) {
      valor += valores[i] * rede.pesos[camada][i][idx];
    }

    valor += __float2half(BIAS * __half2float(rede.bias[camada][idx]));

    //ReLu
    if (__half2float(valor) < 0.f){
      valor = 0.0;
    }

    __syncthreads();
    valores[idx] = valor;

  }

  /**
   * @brief calcula os valores nos neuronios de saida e da uma retorno, utilizar no mesmo kernel de inferencia();
   * 
   * @param retorno ponteiro com o retorno
   * @param valores array de valores
   */
  __device__ void saida(bool* retorno, __half *valores){
    int idx = threadIdx.x;

    if (idx >= NUM_ENTRADAS) return;

    if(idx < NUM_SAIDAS){ // nesse caso só as duas primeiras threads calculam.

      __half valor = 0.0;
    
      for (size_t i = 0; i < NUM_ENTRADAS; i++) {
        valor += valores[i] * rede.saida[idx][i];      
      }

      //ReLu
      if (__half2float(valor) < 0.f){
        retorno[idx] = false;
      }else {
      retorno[idx] = true;
      }
    }
    

  }

  /**
   * @brief muta 10% dos pesos da rede para mais ou para menos dentro do limite de -1 a 1, ideal kernel thread/rede
   * 
   * @param por valor de mutação
   * @param states estados curand para números aleatórios
   */
  __device__ void mutarPesos(__half por, curandState *states){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // muta os pesos de pesos[][][]
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
        for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {

          if (curand_uniform(&states[i]) <= 0.1f) { // 10% chance
            if (curand_uniform(&states[i]) <= 0.5f) { // 50% chance
              rede.pesos[camada][neuronio][peso] += por;
              if (__half2float(rede.pesos[camada][neuronio][peso]) > 1.f) {
                rede.pesos[camada][neuronio][peso] = 1.0;
              }
            }else{
              rede.pesos[camada][neuronio][peso] -= por;
              if (__half2float(rede.pesos[camada][neuronio][peso]) < -1.f) {
                rede.pesos[camada][neuronio][peso] = -1.0;
              }
            }
          }
        }
      }
    }

    //muta pesos de bias[][]
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {

        if (curand_uniform(&states[i]) <= 0.1f) { // 10% chance
          if (curand_uniform(&states[i]) <= 0.5f) { // 50% chance
            rede.bias[camada][peso] += por;
            if (__half2float(rede.bias[camada][peso]) > 1.f) {
              rede.bias[camada][peso] = 1.0;
            }
          }else{
            rede.bias[camada][peso] -= por;
            if (__half2float(rede.bias[camada][peso]) < -1.f) {
              rede.bias[camada][peso] = -1.0;
            }
          }
        }
      }
    }

    //muta os pesos de saida[][]
    for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {

        if (curand_uniform(&states[i]) <= 0.1f) { // 10% chance
          if (curand_uniform(&states[i]) <= 0.5f) { // 50% chance
            rede.saida[neuronio][peso] += por;
            if (__half2float(rede.saida[neuronio][peso]) > 1.f) {
              rede.saida[neuronio][peso] = 1.0;
            }
          }else{
            rede.saida[neuronio][peso] -= por;
            if (__half2float(rede.saida[neuronio][peso]) < -1.f) {
              rede.saida[neuronio][peso] = -1.0;
            }
          }
        }
      }
    }
  }

  /**
   * @brief Verifica qual foi a decisão do individuo na rodada enterior
   * 
   * @return Decisao uma enum com 3 valores possiveis
   */
  __device__ Decisao comprouOuVendeu(){
    if (resultado[0] == true && resultado[1] == false) {
      return Decisao::COMPROU;
    }else if(resultado[0] == false && resultado[1] == true){
      return Decisao::VENDEU;
    }else{
      return Decisao::NADA;
    }
  }
};