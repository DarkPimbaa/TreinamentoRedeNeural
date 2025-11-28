#include <cstddef>
#include <curand_kernel.h>
#include <cstdlib>
#include <ctime>

#define NUM_ENTRADAS 10
#define NUM_CAMADAS 10 // NUM_CAMADAS = camada de entrada + camadas ocultas
#define NUM_SAIDAS 2


struct Neuronio
{
    float valor = 0.f;
    float peso[NUM_ENTRADAS + 1] = { 0.f };
};

struct Camada{
    Neuronio neuronio[NUM_ENTRADAS + 1]; // + 1 é o neuronio de bias
};

struct Rede{
    Camada camada[NUM_CAMADAS];
    bool saida[NUM_SAIDAS] = { false }; // por padrão false
};

// importante usar os defines
class RedeNeural
{
    
    public:
    Rede rede;

    RedeNeural(){
    };

    ~RedeNeural(){

    };

    __device__ void iniciar(float *entrada){
        float valor = 0.f;
        float retorno = 0.f;

        // for que adiciona os valores aos neuronios de entrada sem alterar o bias
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++){
            rede.camada[0].neuronio[neuronio].valor = entrada[neuronio];
        };

        // inicia todos os bias com valor 1.f
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            rede.camada[camada].neuronio[NUM_ENTRADAS].valor = 1.f;
        };

        // for que começa da primeira camada oculta, e faz a magica
        for (size_t camada = 1; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                valor = 0.f;
                // zera o valor do neuronio
                rede.camada[camada].neuronio[neuronio].valor = 0.f;
                // for que multipliva o valor de cada neuronio anterior pelo peso + bias
                for (size_t prev_n = 0; prev_n < NUM_ENTRADAS + 1; prev_n++) {
                   valor += rede.camada[camada - 1].neuronio[prev_n].valor * rede.camada[camada - 1].neuronio[prev_n].peso[neuronio];
                };

                // função de ativação
                if (valor < 0.f) {
                    valor = 0.f;
                };

                rede.camada[camada].neuronio[neuronio].valor = valor;
            };
        };

        //for que vai iterar sobre a ultima camada para gerar a saida
        for (size_t saida = 0; saida < NUM_SAIDAS; saida++) {
            retorno = 0.f;
                for (size_t neuronio = 0; neuronio < NUM_ENTRADAS + 1; neuronio++) {
                
                    retorno += rede.camada[NUM_CAMADAS - 1].neuronio[neuronio].valor * rede.camada[NUM_CAMADAS - 1].neuronio[neuronio].peso[saida];
                
                }

            // função de ativação
            if (retorno < 0.f) {
                rede.saida[saida] = false;
            }else {
                rede.saida[saida] = true;
            };
        };
        
    };

    // muta os pesos da rede pelo valor informado em por
    __device__ void mutacao(float por, curandState *state){

        for (int camada = 0; camada < NUM_CAMADAS; ++camada)
        {
            for (int neuronio = 0; neuronio < NUM_ENTRADAS + 1; ++neuronio)
            {
                for (int peso_idx = 0; peso_idx < NUM_ENTRADAS + 1; ++peso_idx)
                {
                    // 50% de chance de NÃO alterar este peso específico
                    if (curand_uniform(state) < 0.5f)
                        continue;

                    // 50% soma, 50% subtrai o valor 'por'
                    if (curand_uniform(state) < 0.5f)
                        rede.camada[camada].neuronio[neuronio].peso[peso_idx] += por;
                    else
                        rede.camada[camada].neuronio[neuronio].peso[peso_idx] -= por;
                };
            };
        };
    };

    // gera os pesos iniciais no device
    __device__ void gerarPesosIniciais(curandState *state){
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++)
        {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS + 1; neuronio++)
            {
                for (size_t peso = 0; peso < NUM_ENTRADAS + 1; peso++)
                {
                    rede.camada[camada].neuronio[neuronio].peso[peso] = random_device_signed(state);
                }
                
            }
            
        }
        
    }

    // gera um número aleatorio entre -1.f e 1.f no device
    __device__ float random_device_signed(curandState *state) {
        float r = curand_uniform(state);   // (0,1]
        float s = curand_uniform(state);   // decide sinal
        if (s < 0.5f) r = -r;
        return r;
    }

    // gera um número aleatorio entre 0.f e 1.f no device
    __device__ float random_device(curandState *state) {
        return curand_uniform(state); // retorna (0,1]
    }

    // sobreescreve na rede desse objeto
    bool setRede(Rede& rede){
        this->rede = rede;
        return true;
    };

    // retorna a rede completa
    Rede getRede(){
        return this->rede;
    }
};
