#include <cstdio>
#include <ctime>
#include <curand_kernel.h>
#include "./utils.hpp"
#include <stdlib.h>
#include "./types.hpp"

#define NUM_ENTRADAS 60 // 60 = ultimos 10 candles
#define NUM_CAMADAS 6 // NUM_CAMADAS = camada de entrada + camadas ocultas
#define NUM_SAIDAS 2


/** função que normaliza os dados do bitcoin
* @param abertura abertura do Candle
* @param minima minima do Candle
* @param maxima maxima do Candle
* @param fechamento fechamento do Candle
* @return uma struct Candle com os valores normalizados
 * @note chamavel pelo host e pelo device
 */
 __host__ __device__ Candle normalizarBTC(Candle candle){
     float factor = candle.abertura;
     Candle btc;
     
     btc.abertura = candle.abertura / factor;
     btc.maxima = candle.maxima / factor;
     btc.minima = candle.minima / factor;
     btc.fechamento = candle.fechamento / factor;
     btc.volume = candle.volume;
     btc.trades = candle.trades;
     
     return btc;
     
    };

struct Neuronio{
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

//TODO otimizar processamento desperdiçado e tamanho desnecessario

// importante usar os defines
class RedeNeural{
    
    public:
    Rede rede;

    RedeNeural(){
    };

    ~RedeNeural(){

    };

    // inicia os calculos da rede, precisa de um array de float e apenas valores float.
    __device__ void iniciar(float *entrada){
        float valor = 0.f;
        float retorno = 0.f;

        // "zera" o array de saida para evitar falsos positivos
        for (size_t saidaIDX = 0; saidaIDX < NUM_SAIDAS; saidaIDX++) {
            rede.saida[saidaIDX] = false;
        }

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
                // for que multiplica o valor de cada neuronio anterior pelo peso + bias
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

// kernel que recebe o array de redes neurais e o array de Candles
__global__ void kernelTreino(RedeNeural *rede, Candle *candle, int numeroDeCandles, int numeroDeRedes, curandState *states){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numeroDeRedes) {
        curandState local = states[i];
        // TODO adicionar lógica para cada rede ex: o que cada rede vai fazer?
    };
};

__global__ void initStates(curandState *states, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}


int main(){
    // TODO adicionar a logica de ler o csv, e criar o array de redes e o array de Candles - finalizado
    auto h_candles = lerCSV("./filtrado.csv");
    int numeroDeCandles = h_candles.size();
    int numeroDeRedes = 100;
    RedeNeural* h_redes;
    h_redes = new RedeNeural[numeroDeRedes];

    // aloca memoria na gpu e manda os candles
    Candle* d_candles;
    cudaMalloc(&d_candles, numeroDeCandles * sizeof(Candle));
    cudaMemcpy(d_candles, h_candles.data(), numeroDeCandles * sizeof(Candle), cudaMemcpyHostToDevice);
    
    // aloca memoria na gpu e manda as redes
    RedeNeural* d_redes;
    cudaMalloc(&d_redes, numeroDeRedes * sizeof(RedeNeural));
    cudaMemcpy(d_redes, h_redes, numeroDeRedes * sizeof(RedeNeural), cudaMemcpyHostToDevice);

    int block = 32;
    int grid = (numeroDeRedes + block - 1) / block;
    int totalThreads = grid * block;
    curandState* d_states;
    cudaMalloc(&d_states, totalThreads * sizeof(curandState));

    // inicializa os estados
    initStates<<<grid, block>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    kernelTreino<<<grid, block>>>(d_redes, d_candles, numeroDeCandles, numeroDeRedes, d_states);
    cudaDeviceSynchronize();
   

    //TODO adicionar logica de o que fazer com o resultado que vier da gpu, juntamente com o loop de treino

    cudaFree(d_candles);
    cudaFree(d_redes);
    cudaFree(d_states);

    return 0;
}