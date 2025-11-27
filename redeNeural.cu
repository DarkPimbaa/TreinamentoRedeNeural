#include <curand_kernel.h>
#include <cstdlib>
#include <ctime>

#define NUM_ENTRADAS 10
#define NUM_CAMADAS 10 // NUM_CAMADAS = camada de entrada + camadas ocultas
#define NUM_SAIDAS 2


struct Neuronio
{
    float valor = 0.f;
    float peso[NUM_ENTRADAS] = { 0.f };
};

struct Camada{
    Neuronio neuronio[NUM_ENTRADAS];
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

    // TODO terminar de implementar a função iniciar, retorna um int, se for zero é false, se for 1 é true, 11 = true, true.
    __device__ int iniciar(float *entrada){

    };

    // TODO umplementar, essa função vai modificar os pessos aleatóriamente para cima ou para baixo pelo valor passado em por
    __device__ void mutacao(float por){

    };

    // gera os pesos iniciais no device
    __device__ void gerarPesosIniciais(curandState *state){
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++)
        {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++)
            {
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++)
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
