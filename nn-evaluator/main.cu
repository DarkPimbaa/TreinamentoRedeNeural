#define NUM_ENTRADAS 604 // 100 candles * 6 params + 4 extras
#define NUM_CAMADAS 4
#define NUM_SAIDAS 2
#define BIAS 1.f
#define NUM_INDIVIDUOS 1000 // Testando com > 1024
#define META_TAXA_VITORIA 60
#define MAXIMO_RODADAS_SEM_JOGAR 100
#define QUANTOS_CANDLES_POR_GERACAO 2'000
#define PORMIN 0.01 // Taxa de mutação minima
#define PORMAX 0.2 // Taxa de mutação maxima


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

struct Entrada{
    int indiceDaEntrada;
    Decisao decisao;
};
#ifndef __CUDA_ARCH__
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Entrada, indiceDaEntrada, decisao);
#endif
struct Backlog{
    int idxAtual = 0;
    int quantidadeDeEntradas = 0;
    Entrada entradas[1000];
};
#ifndef __CUDA_ARCH__
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Backlog, idxAtual, quantidadeDeEntradas, entradas);
#endif

//* KERNELS =======================================


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

    // verifica se o individuo ganhou na rodada anterior e adiciona a pontuação a ele se estiver vivo, block/rede
    __global__ void kverificarDecisao(RedeNeural * d_individuos, Candle *d_candles, int indiceAtual, Backlog *d_backlog){
        int idx = threadIdx.x;
        if (idx == 0) {
            
                Decisao r = d_individuos[idx].comprouOuVendeu();

                switch (r) {
                    case Decisao::COMPROU:
                        if (d_candles[indiceAtual].fechamento < d_candles[indiceAtual + 2].fechamento) {
                            // ganhou
                            d_individuos[idx].ganho++;
                            d_individuos[idx].partidaSemJogar = 0;
                            d_backlog->quantidadeDeEntradas++;
                            d_backlog->entradas[d_backlog->idxAtual].decisao = Decisao::COMPROU;
                            d_backlog->entradas[d_backlog->idxAtual].indiceDaEntrada = indiceAtual;
                            d_backlog->idxAtual++;
                        }else {
                            d_individuos[idx].perda++;
                            d_individuos[idx].partidaSemJogar = 0;
                            d_backlog->quantidadeDeEntradas++;
                            d_backlog->entradas[d_backlog->idxAtual].decisao = Decisao::COMPROU;
                            d_backlog->entradas[d_backlog->idxAtual].indiceDaEntrada = indiceAtual;
                            d_backlog->idxAtual++;
                        }
                    break;

                    case Decisao::VENDEU:
                        if (d_candles[indiceAtual].fechamento > d_candles[indiceAtual + 2].fechamento) {
                            // ganhou
                            d_individuos[idx].ganho++;
                            d_individuos[idx].partidaSemJogar = 0;
                            d_backlog->quantidadeDeEntradas++;
                            d_backlog->entradas[d_backlog->idxAtual].decisao = Decisao::VENDEU;
                            d_backlog->entradas[d_backlog->idxAtual].indiceDaEntrada = indiceAtual;
                            d_backlog->idxAtual++;
                        }else {
                            d_individuos[idx].perda++;
                            d_individuos[idx].partidaSemJogar = 0;
                            d_backlog->quantidadeDeEntradas++;
                            d_backlog->entradas[d_backlog->idxAtual].decisao = Decisao::VENDEU;
                            d_backlog->entradas[d_backlog->idxAtual].indiceDaEntrada = indiceAtual;
                            d_backlog->idxAtual++;
                        }
                    break;

                    case Decisao::NADA:
                    break;
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

int main(){

    Backlog *h_backlog = new Backlog();
    Backlog *d_backlog;
    cudaMalloc(&d_backlog, sizeof(Backlog));
    cudaMemcpy(d_backlog, h_backlog, sizeof(Backlog), cudaMemcpyHostToDevice);

    Candle* h_candles = new Candle[964750];
    lerCSV_malloccNovo("./btc_windicadores.csv", h_candles, 964750);
    Candle* d_candles;
    cudaMalloc(&d_candles, sizeof(Candle) * 964750);
    cudaMemcpy(d_candles, h_candles, sizeof(Candle) * 964750, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    __half* h_valores = new __half[NUM_ENTRADAS];
    __half* d_valores;
    cudaMalloc(&d_valores, sizeof(float) * NUM_ENTRADAS);
    
    // cria os individuos e inicia os pesos
    RedeNeural* h_individuos = new RedeNeural[2];
    h_individuos[0].init();
    h_individuos[1].init();
    h_individuos[0].rede = carregarRedeJSON("./melhor.json");
    h_individuos[1].rede = carregarRedeJSON("./melhor.json");
    RedeNeural* d_individuos;
    cudaMalloc(&d_individuos, sizeof(RedeNeural) * 2);
    cudaMemcpy(d_individuos, h_individuos, sizeof(RedeNeural) * 2, cudaMemcpyHostToDevice);

    int numCandlesHist = (NUM_ENTRADAS - 4) / 6;
        int startRodada = numCandlesHist + 10; // margem de segurança // com 604 entradas começa no candle 110
        if (startRodada < 20) startRodada = 20;

        for (size_t rodada = startRodada; rodada < QUANTOS_CANDLES_POR_GERACAO; rodada++) {
            knormalizar<<<1,32>>>(d_candles, rodada, d_valores);
            cudaDeviceSynchronize(); 
            kinferencia<<<1, NUM_ENTRADAS>>>(d_valores, d_individuos);
            cudaDeviceSynchronize();

            kverificarDecisao<<<1, 1>>>(d_individuos, d_candles, rodada, d_backlog);
            cudaDeviceSynchronize();
        }

    cudaMemcpy(h_backlog, d_backlog, sizeof(Backlog), cudaMemcpyDeviceToHost);

    #ifndef __CUDA_ARCH__
    json j = *h_backlog;
    std::ofstream f("backlog.json");
    f << j.dump(4);
    #endif
    
    return 0;
}