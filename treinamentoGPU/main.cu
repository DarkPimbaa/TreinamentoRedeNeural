#define NUM_ENTRADAS 64 // 100 candles * 6 params + 4 extras
#define NUM_CAMADAS 4
#define NUM_SAIDAS 2
#define BIAS 1.f
#define NUM_INDIVIDUOS 1000 // Testando com > 1024
#define META_TAXA_VITORIA 60
#define MAXIMO_RODADAS_SEM_JOGAR 100
#define QUANTOS_CANDLES_POR_GERACAO 2'000
#define PORMIN 0.01 // Taxa de mutação minima
#define PORMAX 0.5 // Taxa de mutação maxima


//* INCLUDES
#include "./includes/redeNeural.cu"
#include <cstdio>
#include <ctime>
#include "./includes/types.hpp"
#include "./includes/utils.hpp"
#include <cuda_fp16.h>


#ifndef __CUDA_ARCH__
#include <nlohmann/json.hpp>
#include <fstream>
using json = nlohmann::json;
#endif

void salvarRedeJSON(const Rede& r, const char* arquivo) {
#ifndef __CUDA_ARCH__
    json j;

    for (int c = 0; c < NUM_CAMADAS; ++c)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            for (int k = 0; k < NUM_ENTRADAS; ++k)
                j["pesos"][c][i][k] = __half2float(r.pesos[c][i][k]);

    for (int c = 0; c < NUM_CAMADAS; ++c)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            j["bias"][c][i] = __half2float(r.bias[c][i]);

    for (int s = 0; s < NUM_SAIDAS; ++s)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            j["saida"][s][i] = __half2float(r.saida[s][i]);

    std::ofstream out(arquivo);
    out << j.dump(2);
#endif
}

Rede carregarRedeJSON(const char* arquivo) {
    Rede r{};

#ifndef __CUDA_ARCH__
    std::ifstream in(arquivo);
    json j;
    in >> j;

    for (int c = 0; c < NUM_CAMADAS; ++c)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            for (int k = 0; k < NUM_ENTRADAS; ++k)
                r.pesos[c][i][k] = __float2half(j["pesos"][c][i][k].get<float>());

    for (int c = 0; c < NUM_CAMADAS; ++c)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            r.bias[c][i] = __float2half(j["bias"][c][i].get<float>());

    for (int s = 0; s < NUM_SAIDAS; ++s)
        for (int i = 0; i < NUM_ENTRADAS; ++i)
            r.saida[s][i] = __float2half(j["saida"][s][i].get<float>());
#endif

    return r;
}




//* KERNELS ///////////////////////////////////////////////////////////////////////////////////////////

    __global__ void init_curand(curandState *states, unsigned long seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_INDIVIDUOS) {
        // seed      -> mesma base para todos
        // sequence  -> diferente por thread
        // offset    -> 0 normalmente
        curand_init(seed, i, 0, &states[i]);
    }
    };

    __global__ void kiniciarPesos(curandState* states, RedeNeural* d_individuos){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < NUM_INDIVIDUOS) {
            d_individuos[idx].iniciarPesos(states);
        }
    };

    // recebe os valores, faz a inferencia e coloca o resultado no array de resultados block/rede
    __global__ void kinferencia(__half *d_valores, RedeNeural *d_individuos){
        int idx = threadIdx.x;
        int block = blockIdx.x; // Block ID corresponds to the individual
        
        // Ensure we don't access memory out of bounds if configured incorrectly
        if (block >= NUM_INDIVIDUOS) return;

        __shared__ bool retorno[NUM_SAIDAS];
        __shared__ __half Svalores[NUM_ENTRADAS];

        if (idx < NUM_ENTRADAS) {
            Svalores[idx] = d_valores[idx];
        }
        __syncthreads();

        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            d_individuos[block].inferencia(Svalores, camada);
        }

        if (idx < NUM_SAIDAS) {
            d_individuos[block].saida(retorno, Svalores);
        }
        __syncthreads();

        if (idx < NUM_SAIDAS) {
            d_individuos[block].resultado[idx] = retorno[idx];
        }

    };

    // muta os pesos das redes thread/rede
    __global__ void kmutarPesos(RedeNeural *d_individuos, __half por, curandState *d_states){ 
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < NUM_INDIVIDUOS) {
            d_individuos[idx].mutarPesos(por, d_states);
        }
    };

    // verifica se o individuo ganhou na rodada anterior e adiciona a pontuação a ele se estiver vivo, thread/rede
    __global__ void kverificarDecisao(RedeNeural * d_individuos, Candle *d_candles, int indiceAtual){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < NUM_INDIVIDUOS) {
            if (d_individuos[idx].bvivo == true) {
            
                Decisao r = d_individuos[idx].comprouOuVendeu();

                switch (r) {
                    case Decisao::COMPROU:
                        if (d_candles[indiceAtual].fechamento < d_candles[indiceAtual + 1].fechamento) {
                            // ganhou
                            d_individuos[idx].ganho++;
                            d_individuos[idx].partidaSemJogar = 0;
                        }else {
                            d_individuos[idx].perda++;
                            d_individuos[idx].partidaSemJogar = 0;
                        }
                    break;

                    case Decisao::VENDEU:
                        if (d_candles[indiceAtual].fechamento > d_candles[indiceAtual + 1].fechamento) {
                            // ganhou
                            d_individuos[idx].ganho++;
                            d_individuos[idx].partidaSemJogar = 0;
                        }else {
                            d_individuos[idx].perda++;
                            d_individuos[idx].partidaSemJogar = 0;
                        }
                    break;

                    case Decisao::NADA:
                        d_individuos[idx].partidaSemJogar++;
                        if (d_individuos[idx].partidaSemJogar >=MAXIMO_RODADAS_SEM_JOGAR) {
                            d_individuos[idx].bvivo = false;
                        }
                    break;
                }
            }
        }
    }

    // normaliza os valores dos candles <<1, 32>>
    __global__ void knormalizar(Candle *d_candles, int indiceAtual, __half *d_valores){
        int idx = threadIdx.x;

        if (idx == 0) {
        
            float factor = d_candles[indiceAtual].fechamento;
            float factorVolume = d_candles[indiceAtual].volume;
            float factorTrades = d_candles[indiceAtual].trades;
            
            // Calculando dinamicamente quantos candles históricos precisamos com base no NUM_ENTRADAS
            // (NUM_ENTRADAS - 4 globais) / 6 parâmetros por candle
            int numCandlesHist = (NUM_ENTRADAS - 4) / 6;

            int idxInicio = indiceAtual - numCandlesHist;
            int ii = 0;
            for (size_t i = idxInicio + 1; i < indiceAtual + 1; i++) {
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].abertura / factor);
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].maxima / factor);
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].minima / factor);
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].fechamento / factor);
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].volume / factorVolume);
                if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[i].trades / factorTrades);
            }

            if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[indiceAtual].mm7 / factor);
            if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[indiceAtual].mm21 / factor);
            if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)(d_candles[indiceAtual].mm50 / factor);
            if (ii < NUM_ENTRADAS) d_valores[ii++] = (__half)d_candles[indiceAtual].rsi14;
        }
        __syncthreads();
    };

    // Encontra o melhor indivíduo (roda em 1 thread apenas para simplicidade ou redução)
    __global__ void kEncontrarMelhor(RedeNeural *d_individuos, bool *d_brun, RedeNeural *d_melhor) {
        int idx = threadIdx.x;
        if (idx == 0) {
            for (size_t i = 0; i < NUM_INDIVIDUOS; i++) {
                if (d_individuos[i].attTaxa() > d_melhor->taxaVitoria && d_individuos[i].bvivo == true) {
                    *d_melhor = d_individuos[i];
                }
            }

            printf("taxa de vitoria do melhor individuo: %f\n", d_melhor->taxaVitoria);
            printf("total ganho: %i\n", d_melhor->ganho);
            printf("total perda: %i\n", d_melhor->perda);
            
            if (d_melhor->taxaVitoria >= META_TAXA_VITORIA) {
                *d_brun = false;
                printf("Meta atingida!");
            }
        }
    }

    // Propaga o melhor indivíduo para todos os outros
    __global__ void kPropagarMelhor(RedeNeural *d_individuos, RedeNeural *d_melhor) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (idx < NUM_INDIVIDUOS) {
            // Cópia direta da memória global para memória global, evitando stack
            d_individuos[idx] = *d_melhor;
            
            // Reset status
            d_individuos[idx].ganho = 0;
            d_individuos[idx].perda = 0;
            d_individuos[idx].partidaSemJogar = 0;
            d_individuos[idx].bvivo = true;
            d_individuos[idx].taxaVitoria = 0.0f;
        }
    }

    

//* FIM KERNEL ///////////////////////////////////////////////////////////////////////////////////////

int main(){


    printf("Iniciando treinamento com:\n");
    printf("Individuos: %d\n", NUM_INDIVIDUOS);
    printf("Entradas: %d\n", NUM_ENTRADAS);

    //inicia os states
    curandState *d_states;
    cudaMalloc(&d_states, sizeof(curandState) * NUM_INDIVIDUOS);
    init_curand<<<(NUM_INDIVIDUOS + 256 - 1) / 256, 256>>>(d_states, time(NULL));
    cudaDeviceSynchronize();

    Candle* h_candles = new Candle[964750];
    lerCSV_malloccNovo("./btc_windicadores.csv", h_candles, 964750);
    Candle* d_candles;
    cudaMalloc(&d_candles, sizeof(Candle) * 964750);
    cudaMemcpy(d_candles, h_candles, sizeof(Candle) * 964750, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    __half* h_valores = new __half[NUM_ENTRADAS];
    __half* d_valores;
    cudaMalloc(&d_valores, sizeof(float) * NUM_ENTRADAS);
    RedeNeural* h_melhor = new RedeNeural();
    h_melhor->init();
    RedeNeural* d_melhor;
    cudaMalloc(&d_melhor, sizeof(RedeNeural));
    cudaMemcpy(d_melhor, h_melhor, sizeof(RedeNeural), cudaMemcpyHostToDevice);
    
    // cria os individuos e inicia os pesos
    RedeNeural* h_individuos = new RedeNeural[NUM_INDIVIDUOS];
    RedeNeural* d_individuos;
    cudaMalloc(&d_individuos, sizeof(RedeNeural) * NUM_INDIVIDUOS);
    cudaMemcpy(d_individuos, h_individuos, sizeof(RedeNeural) * NUM_INDIVIDUOS, cudaMemcpyHostToDevice);
    
    kiniciarPesos<<<1, NUM_INDIVIDUOS>>>(d_states, d_individuos);
    cudaDeviceSynchronize();

    bool* h_brun = new bool(true);
    bool* d_brun;
    cudaMalloc(&d_brun, sizeof(bool));
    cudaMemcpy(d_brun, h_brun, sizeof(bool), cudaMemcpyHostToDevice);

    //* chama o kernelzão de treinamento
    int geracao = 0;
    while(*h_brun){
        printf("Geracao: %i\n", geracao);
        
        // Calculando quantos candles precisamos para trás
        int numCandlesHist = (NUM_ENTRADAS - 4) / 6;
        int startRodada = numCandlesHist + 10; // margem de segurança
        if (startRodada < 20) startRodada = 20;

        for (size_t rodada = startRodada; rodada < QUANTOS_CANDLES_POR_GERACAO; rodada++) {
            knormalizar<<<1,32>>>(d_candles, rodada, d_valores);
            cudaDeviceSynchronize(); 
            kinferencia<<<NUM_INDIVIDUOS, NUM_ENTRADAS>>>(d_valores, d_individuos);
            cudaDeviceSynchronize();

            kverificarDecisao<<<1, NUM_INDIVIDUOS>>>(d_individuos, d_candles, rodada);
            cudaDeviceSynchronize();
        }
        
        // Passos de reprodução divididos
        kEncontrarMelhor<<<1, 1>>>(d_individuos, d_brun, d_melhor);
        cudaDeviceSynchronize();
        
        kPropagarMelhor<<<1, 32>>>(d_individuos, d_melhor);
        cudaDeviceSynchronize();

        kmutarPesos<<<1,NUM_INDIVIDUOS>>>(d_individuos, Rand::Float(PORMIN, PORMAX), d_states);
        cudaDeviceSynchronize();

        cudaMemcpy(h_brun, d_brun, sizeof(bool), cudaMemcpyDeviceToHost);
        geracao++;
    }

    cudaMemcpy(h_melhor, d_melhor, sizeof(RedeNeural), cudaMemcpyDeviceToHost);
    salvarRedeJSON(h_melhor->rede, "teste.json");


    //* zona de liberar memoria------------------------
    delete h_brun;
    delete[] h_valores;
    delete[] h_individuos;
    delete[] h_candles;
    delete h_melhor;
    cudaFree(d_melhor);
    cudaFree(d_brun);
    cudaFree(d_valores);
    cudaFree(d_candles);
    cudaFree(d_states);
    cudaFree(d_individuos);
    return 0;
}