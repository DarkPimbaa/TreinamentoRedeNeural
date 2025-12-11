#ifndef NUM_ENTRADAS
#define NUM_ENTRADAS 60
#endif
#ifndef NUM_CAMADAS
#define NUM_CAMADAS 5
#endif
#ifndef NUM_SAIDAS
#define NUM_SAIDAS 2
#endif
#ifndef BIAS
#define BIAS 1.f
#endif

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "./utils.hpp"

// === estrutura de dados

struct Neuronio{
    float valor = 0.f;
    __half pesos[NUM_ENTRADAS] = {0.f};
};

struct Camada {
    Neuronio neuronio[NUM_ENTRADAS];
};

struct Saida{
    Neuronio neuronio[NUM_SAIDAS];
};

struct Rede {
    float entrada[NUM_ENTRADAS] = {0.f};
    Camada oculto[NUM_CAMADAS];
    Saida saida;
};

// === final da estrutura de dados


// rede neural
class RedeNeural{
public:    
    Rede rede;
    int ganho = 0;
    int perda = 0;
    bool retorno[NUM_SAIDAS] = {false};

    __host__ RedeNeural(){
        //iniciarPesos();
    };

    // inicia os pesos com valores aleatorios
    __host__ void iniciarPesos(){

        // muta os pesos das camadas ocultas
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                    
                    rede.oculto[camada].neuronio[neuronio].pesos[peso] = __float2half(Rand::Float(-1.f,1.f));         
                    
                }
            }
        }

        // muta os pesos da camada de saida
        for (size_t saida = 0; saida < NUM_SAIDAS; saida++) {
            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                
                rede.saida.neuronio[saida].pesos[peso] = __float2half(Rand::Float(-1.f,1.f));        
                
            }
        }

    }

    // inicia os calculos da rede e retorna um vetor de boleanos com o resultado
    __device__ void iniciar(float* valores){


        // inicia os valores nos neuronios de entrada
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
            rede.entrada[neuronio] = valores[neuronio];
        }

        // zera o vetor de retorno
        for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
            retorno[neuronio] = false;
        }


        // alimenta a primeira camada oculta
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {

            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                
                rede.oculto[0].neuronio[neuronio].valor += rede.entrada[peso] * __half2float(rede.oculto[0].neuronio[neuronio].pesos[peso]);

            }

            // aplica o BIAS
            rede.oculto[0].neuronio[neuronio].valor += BIAS;

            // ReLu (função de ativação)
            if (rede.oculto[0].neuronio[neuronio].valor < 0.f) {
                rede.oculto[0].neuronio[neuronio].valor = 0.f;
            }
        }

        // loop do restante das camadas ocultas
        for (size_t camada = 1; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                    rede.oculto[camada].neuronio[neuronio].valor += rede.oculto[camada - 1].neuronio[peso].valor * __half2float(rede.oculto[camada].neuronio[neuronio].pesos[peso]); 
                }

                //aplica o bias
                rede.oculto[camada].neuronio[neuronio].valor += BIAS;
                
                //ReLu
                if (rede.oculto[camada].neuronio[neuronio].valor < 0.f) {
                    rede.oculto[camada].neuronio[neuronio].valor = 0.f;
                }
            }
        }


        // loop camadas de saida
        for (size_t saida = 0; saida < NUM_SAIDAS; saida++) {
            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                rede.saida.neuronio[saida].valor += rede.oculto[NUM_CAMADAS - 1].neuronio[peso].valor * __half2float(rede.saida.neuronio[saida].pesos[peso]);
            }

            // aplica bias
            rede.saida.neuronio[saida].valor += BIAS;

            // ReLU
            if (rede.saida.neuronio[saida].valor < 0.f) {
                rede.saida.neuronio[saida].valor = 0.f;
            }

            // alimenta retorno
            if (rede.saida.neuronio[saida].valor == 0) {
                this->retorno[saida] = false;
            }else {
                this->retorno[saida] = true;
            }
        }

        clearValores();
    };

    // tem 10% de chance de mutar um peso aleatoriamente para cima ou para baixo limitado entre -1.f e 1.f
    __host__ void mutacao(float por = 0.1f){

        // loop para as ocultas
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                    
                    // 10% de chance
                    if (Rand::Int(1, 10) == 1) {

                        // 50% de chance
                        if (Rand::Int(0,1) == 1) {
                            rede.oculto[camada].neuronio[neuronio].pesos[peso] += __float2half(por);

                            if (rede.oculto[camada].neuronio[neuronio].pesos[peso] > __float2half(1.f)) {
                                rede.oculto[camada].neuronio[neuronio].pesos[peso] = __float2half(1.f);
                            }
                        }else {
                            rede.oculto[camada].neuronio[neuronio].pesos[peso] -= __float2half(por);

                            if (rede.oculto[camada].neuronio[neuronio].pesos[peso] < __float2half(-1.f)) {
                                rede.oculto[camada].neuronio[neuronio].pesos[peso] = __float2half(-1.f);
                            }
                        }
                    }
                }
            }
        }

        // loop para os pesos de neuronios de saida
        for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                
                // 10% chance
                if (Rand::Int(0,10) == 1) {
                    
                    // 50% chance
                    if (Rand::Int(0,1) == 1) {
                        rede.saida.neuronio[neuronio].pesos[peso] += __float2half(por);
                        
                        if (rede.saida.neuronio[neuronio].pesos[peso] > __float2half(1.f)) {
                            rede.saida.neuronio[neuronio].pesos[peso] = __float2half(1.f);
                        }
                    }else {
                        rede.saida.neuronio[neuronio].pesos[peso] -= __float2half(por);

                        if (rede.saida.neuronio[neuronio].pesos[peso] < __float2half(-1.f)) {
                            rede.saida.neuronio[neuronio].pesos[peso] = __float2half(-1.f);
                        }
                    }
                
                }
            }
        }
    }

    // zera os valores de toda a rede
    __device__ void clearValores(){


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
            rede.saida.neuronio[neuronio].valor = 0.f;
        }


    };

    void setRede(Rede rede){
        this->rede = rede;
    };

    Rede getRede(){
        return this->rede;
    }
};