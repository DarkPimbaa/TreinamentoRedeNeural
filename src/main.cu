#define NUM_INDIVIDUOS 2000
#define NUM_CAMADAS 4
#define NUM_NEURONIOS 608 // multiplo de 16
#define NUM_SAIDAS 2
#define TEMPO_TREINAMENTO 60.0 // tempo total de treinamento em minutos
#define NUM_CANDLES_BATCH 20'000 // numero de candles que cada individuo vai "apostar" antes da verificação
#define TAMANHO_BTCCSV 954620

#define loopGeracao(x) while(x)


#include <ctime>
#include <cstddef>
#include <cstdio>
#include "../include/types.hpp"
#include "../include/utils.hpp"
#include <cstdlib>
#include "../include/individuo.cuh"
#include <cublas_v2.h>

int main() {

    // carrega os candles
    Candle* h_candles = (Candle*)malloc(TAMANHO_BTCCSV * sizeof(Candle));
    lerCSV_mallocc("btc.csv", h_candles, TAMANHO_BTCCSV);
    Candle* d_candles;
    cudaMalloc(&d_candles, (TAMANHO_BTCCSV * sizeof(Candle)));
    cudaMemcpy(d_candles, h_candles, TAMANHO_BTCCSV * sizeof(Candle), cudaMemcpyHostToDevice);
    
    // cria os individuos na vram
    Individuo* h_individuos = new Individuo[NUM_INDIVIDUOS];
    for (size_t i = 0; i<NUM_INDIVIDUOS; i++) {
        h_individuos[i].init();
    }
    Individuo* d_individuos;
    cudaMalloc(&d_individuos, sizeof(Individuo) * NUM_INDIVIDUOS);
    cudaMemcpy(d_individuos, h_individuos, sizeof(Individuo) * NUM_INDIVIDUOS, cudaMemcpyHostToDevice);

    //crie os pesos
    IndividuosPesos* h_pesos = new IndividuosPesos();
    IndividuosPesos* d_pesos;
    cudaMalloc(&d_pesos, sizeof(IndividuosPesos));
    cudaMemcpy(d_pesos, h_pesos, sizeof(IndividuosPesos),cudaMemcpyHostToDevice);

    IniciarEstadosCurand<<<getblocksize(NUM_INDIVIDUOS, 256), 256>>>(d_individuos, time(NULL));
    cudaDeviceSynchronize();

    IniciarPesosEbias<<<getblocksize(NUM_INDIVIDUOS, 256), 256>>>(d_individuos, d_pesos);
    cudaDeviceSynchronize();

    int* d_idxMelhor;
    cudaMalloc(&d_idxMelhor, sizeof(int));
    cudaMemset(d_idxMelhor, 0, sizeof(int)); // Initialize to 0, though logic handles it.
    
    bool brun = true;
    double tempoTotal = 0;
    int geracao = 0;
    float rangeRandom = NUM_NEURONIOS*NUM_NEURONIOS;
    // loop de geração
    loopGeracao(brun) {

        printf("Geracao: %d\n", geracao);
        printf("Tempo decorrido: %.2f\n", tempoTotal);
        Timer t(TimeUnit::m);

        // verifica tempo total, se for maior ou igual ao tempo limite de treino, para.
        if (tempoTotal >= TEMPO_TREINAMENTO) {
            brun = false;
        }

        // entra no loop de inferecia candle x até candle x
        for (size_t i = NUM_NEURONIOS / 6; i < NUM_CANDLES_BATCH; i += CANDLE_BATCH_SIZE) {
            
            int numCandles = CANDLE_BATCH_SIZE;
            if (i + numCandles > NUM_CANDLES_BATCH) {
                numCandles = NUM_CANDLES_BATCH - i;
            }

            if (i % 1000 < CANDLE_BATCH_SIZE) { // Improved logging frequency check
                cudaDeviceSynchronize();
                printf("Processando Candle: %lu / %d\r", i, NUM_CANDLES_BATCH);
                fflush(stdout);
            }

            // kernel que normaliza valores
            dim3 dimBlockNorm(256);
            dim3 dimGridNorm(getblocksize(NUM_NEURONIOS, 256), numCandles);
            normalizarValores<<<dimGridNorm, dimBlockNorm>>>(d_pesos, d_candles, i, numCandles);

            inferencia(d_individuos, d_pesos, NUM_CAMADAS, numCandles);
            
            // kernel que verifica saida e aplica pontuação;
            dim3 dimBlockVer(256);
            dim3 dimGridVer(getblocksize(NUM_INDIVIDUOS, 256), numCandles);
            verificarCompraVenda<<<dimGridVer, dimBlockVer>>>(d_individuos, d_pesos, d_candles, i, numCandles);
        }
        cudaDeviceSynchronize();

        // kernel que verifica qual for o melhor
        verificarMelhor<<<1,32>>>(d_individuos, d_idxMelhor, NUM_CANDLES_BATCH);
        cudaDeviceSynchronize();

        // kernel que repopular o array de individuos com copias do melhor
        ClonarDNAMassa<<<getblocksize(NUM_NEURONIOS * NUM_NEURONIOS, 256), 256>>>(d_pesos, d_idxMelhor);
        cudaDeviceSynchronize();
        // kernel que muta os pesos de todos os individuos exerto melhor
        AplicarMutacaoGlobal<<<getblocksize(NUM_INDIVIDUOS, 256), 256>>>(d_individuos, d_pesos, d_idxMelhor, rangeRandom);
        cudaDeviceSynchronize();

        //zera os valores dos individuos
        zeraIndividuos<<<getblocksize(NUM_INDIVIDUOS, 256),256>>>(d_individuos);
        cudaDeviceSynchronize();

        rangeRandom = rangeRandom*0.99;
        if (rangeRandom < NUM_NEURONIOS * 10) {
            rangeRandom = NUM_NEURONIOS * 10;
        }
        tempoTotal += t.stop();
        geracao++;
    } 

    cudaFree(d_candles);
    cudaFree(d_individuos);
    
    return 0;
}
