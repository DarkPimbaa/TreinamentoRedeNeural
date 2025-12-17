#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#ifndef NUM_ENTRADAS
#define NUM_ENTRADAS 60 // 60 floats, 60 neuronios + 1 bias;
#endif
#ifndef NUM_CAMADAS
#define NUM_CAMADAS 4
#endif
#ifndef NUM_SAIDAS
#define NUM_SAIDAS 2
#endif

struct Neuronio{
  float valor = 0.f;
  __half pesos[NUM_ENTRADAS] = {0};
};

struct Bias{
  const float valor = 1.f;
  __half pesos[NUM_ENTRADAS];
};

struct Camada{
  Neuronio neuronio[NUM_CAMADAS];
  Bias BIAS;
};

struct Rede{
  float entrada[NUM_ENTRADAS] = {0.f};
  Camada oculto[NUM_CAMADAS];
  Neuronio saida[NUM_SAIDAS];
  bool retorno[NUM_SAIDAS];
};

// rede neural
class RedeNeural {
public:
  Rede rede;
  int ganho = 0;
  int perda = 0;
  bool bvivo = true;
  bool retorno[NUM_SAIDAS] = {false};
  int rodadasSemApostar = 0;

  __host__ RedeNeural() {
    // iniciarPesos();
  };

  __device__ float randDevice(curandState *state) {
    // número aleatório uniformemente distribuído entre -1 e 1
    return curand_uniform(state) * 2.f - 1.f;
  }

  // inicia os pesos da rede incluindo BIAS
  __device__ void iniciarPesosDevice(Rede &rede, curandState *estados) {
    
    // inicia os pesos das camadas ocultas + BIAS
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++){
        for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
          rede.oculto[camada].neuronio[neuronio].pesos[peso] = __float2half(randDevice(estados));
          rede.oculto[camada].BIAS.pesos[peso] = __float2half(randDevice(estados));
        }
      }
    }


    // inicia os pesos da camada de saida
    for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++){
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.saida[neuronio].pesos[peso] = __float2half(randDevice(estados));
      }
    }

  }

  // função que alimenta a primeira camada oculta
  __device__ void inferenciaPrimeiraOculta(float* valoresNormalizados){
    int idx = threadIdx.x;

      // cada neuronio calcula seu valor
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.oculto[0].neuronio[idx].valor += rede.entrada[peso] * __half2float(rede.oculto[0].neuronio[idx].pesos[peso]);
      }

      rede.oculto[0].neuronio[idx].valor += rede.oculto[0].BIAS.valor * __half2float(rede.oculto[0].BIAS.pesos[idx]);

      //ReLu
      if (rede.oculto[0].neuronio[idx].valor < 0.f) {
        rede.oculto[0].neuronio[idx].valor = 0.f;
      }
  }

    // função chamada por camada, cada thread cuida de um neuronio. Utilizar em um loop de kernels(que vai iterar para cada camada) que chama um kernel que executa essa função
  __device__ void inferenciaOculta(int camada = 1){
    int idx = threadIdx.x;

      // cada neuronio calcula seu valor
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.oculto[camada].neuronio[idx].valor += rede.oculto[camada - 1].neuronio[peso].valor * __half2float(rede.oculto[camada].neuronio[idx].pesos[peso]);
      }

      rede.oculto[camada].neuronio[idx].valor += rede.oculto[camada].BIAS.valor * __half2float(rede.oculto[camada].BIAS.pesos[idx]);

      //ReLu
      if (rede.oculto[camada].neuronio[idx].valor < 0.f) {
        rede.oculto[camada].neuronio[idx].valor = 0.f;
      }
  }

  // faz a inferencia na camada de saida e já alimento o retorno e já cleana valores.
  __device__ void inferenciaSaida(){

    for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        rede.saida[neuronio].valor += rede.oculto[NUM_CAMADAS - 1].neuronio[peso].valor * __half2float(rede.saida[neuronio].pesos[peso]);
      }

      //ReLu
      if (rede.saida[neuronio].valor < 0.f) {
        retorno[neuronio] = false;
      }else {
        retorno[neuronio] = true;
      }

    }

    clearValores();


  }
  
  // tem 10% de chance de mutar um peso aleatoriamente para cima ou para baixo, limitado entre -1.f e 1.f. Usar dentro de um kernel que vai iterar em cada camada
  __device__ void mutacaoDeviceOculta(float por, curandState *state, int camada) {
    int idx = threadIdx.x;

    for (size_t i = 0; i < NUM_ENTRADAS; i++) {
      float chance = curand_uniform(state);
      if (chance <= 0.10f) {
        if (chance <=0.50f) {
          rede.oculto[camada].neuronio[idx].pesos[i] += __float2half(por);
        }else{
          rede.oculto[camada].neuronio[idx].pesos[i] -= __float2half(por);
        }

        if (rede.oculto[camada].neuronio[idx].pesos[i] < __float2half(-1.f)) { // se for menor que -1 = 1
          rede.oculto[camada].neuronio[idx].pesos[i] = __float2half(-1.f);
        }else if(rede.oculto[camada].neuronio[idx].pesos[i] > __float2half(1.f)){ // se for maior que 1 = 1
          rede.oculto[camada].neuronio[idx].pesos[i] = __float2half(1.f);
        }
      }
    }
  }

  // muta os pesos de todos os BIAS das camadas ocultas
  __device__ void mutarBias(float por, curandState* state){
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        float chance = curand_uniform(state);
        if (chance <= 0.10f) {
          //muta
          if (chance <= 0.50f) {
            rede.oculto[camada].BIAS.pesos[peso] += __float2half(por);
          }else {
            rede.oculto[camada].BIAS.pesos[peso] -= __float2half(por);
          }

          // limite
          if (rede.oculto[camada].BIAS.pesos[peso] > __float2half(1.0)) {
            rede.oculto[camada].BIAS.pesos[peso] = __float2half(1.f);
          }else if(rede.oculto[camada].BIAS.pesos[peso] < __float2half(-1.f)){
            rede.oculto[camada].BIAS.pesos[peso] = __float2half(-1.f);
          }
        }
      }
    }
  }

  // muta os pesos dos neuronios de saida
  __device__ void mutarSaida(float por, curandState* state){
    for (size_t neuronio = 0; neuronio < NUM_CAMADAS; neuronio++) {
      for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
        float chance = curand_uniform(state);
        if (chance <= 0.10f) {
          //muta
          if (chance <= 0.50f) {
            rede.saida[neuronio].pesos[peso] += __float2half(por);
          }else {
            rede.saida[neuronio].pesos[peso] -= __float2half(por);
          }

          // limite
          if (rede.saida[neuronio].pesos[peso] > __float2half(1.0)) {
            rede.saida[neuronio].pesos[peso] = __float2half(1.f);
          }else if(rede.saida[neuronio].pesos[peso] < __float2half(-1.f)){
            rede.saida[neuronio].pesos[peso] = __float2half(-1.f);
          }
        }
      }
    }
  }

  // zera os valores de toda a rede. Cada thread pode fazer isso individualmente
  __device__ void clearValores() {

    // loop de entrada
    for (size_t i = 0; i < NUM_ENTRADAS; i++) {
      rede.entrada[i] = 0.f;
    }

    // loop das camadas ocultas
    for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
      for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
        rede.oculto[camada].neuronio[neuronio].valor = 0.f;
      }
    }

    // loop neuronios de saida
    for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
      rede.saida[neuronio].valor = 0.f;
    }
  };
};