#include <cstdio>
#define NUM_ENTRADAS 60
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2
#define BIAS 1.f
#define NUM_INDIVIDUOS 100 // até 100mil se for 60 entradas e até 2mil se for 600 entradas
#define META_TAXA_VITORIA 60 

#include <cstddef>
#include "includes/redeNeural.cu"
#include "includes/types.hpp"
#include "includes/utils.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>

struct Melhores{
    RedeNeural rede;
    float taxa = 0;
};

__global__ void initCurand(curandState* estados, unsigned long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &estados[tid]);
}

__global__ void treinamento(Candle* d_candles, RedeNeural* d_individuos, int n){
    int individuo = blockIdx.x; 
    int lane      = threadIdx.x;

    if (individuo >= n) return;

    if (lane == 0) {
        int janela = NUM_ENTRADAS / 6; // 10 candles
        float valores[NUM_ENTRADAS];   // array de 60 floats

        // itera sobre todo o dataset
        for (size_t i = (janela - 1); i < 1000/*964799*/; i++) {
            
            // O "factor" é a abertura do candle atual (o mais recente da janela)
            float factor = d_candles[i].abertura;
            
            // Proteção contra divisão por zero
            if (factor < 0.00001f) factor = 1.0f; 

            int idx_val = 0;

            // Loop que reconstrói a janela de 10 candles olhando para trás
            // k=0 pega o candle mais antigo (i - 9)
            // k=9 pega o candle atual (i)
            for (int k = 0; k < janela; k++) {
                
                // Matemática de índice: Se i=100 e janela=10:
                // k=0 -> lê índice 91
                // ...
                // k=9 -> lê índice 100
                int indice_leitura = i - (janela - 1) + k;
                const Candle& c = d_candles[indice_leitura];

                valores[idx_val + 0] = c.abertura   / factor;
                valores[idx_val + 1] = c.maxima     / factor;
                valores[idx_val + 2] = c.minima     / factor;
                valores[idx_val + 3] = c.fechamento / factor;
                
                valores[idx_val + 4] = (float)c.volume;
                valores[idx_val + 5] = (float)c.trades;

                idx_val += 6; 
            }

            // Alimenta a rede neural
            d_individuos[individuo].iniciar(valores);

            // verifica se ganhou ou perdeu
            if(d_individuos[individuo].retorno[0] == true && d_individuos[individuo].retorno[1] == false){
                // verifica se o proximo candle é de alta, se sim ganhou se não perdeu
                if (d_candles[i].fechamento < d_candles[i + 1].fechamento) {
                    d_individuos[individuo].ganho++;
                }else {
                    d_individuos[individuo].perda++;
                }

                d_individuos[individuo].rodadasSemApostar = 0;
            }else if (d_individuos[individuo].retorno[0] == false && d_individuos[individuo].retorno[1] == true) {
                // verifica se o proximo candle é de baixa, se sim ganhou se não perdeu
                if (d_candles[i].fechamento > d_candles[i + 1].fechamento) {
                    d_individuos[individuo].ganho++;
                }else {
                    d_individuos[individuo].perda++;
                }

                d_individuos[individuo].rodadasSemApostar = 0;
            }else {
                // não apostou
                d_individuos[individuo].rodadasSemApostar++;

                if (d_individuos[individuo].rodadasSemApostar >=100) {
                    d_individuos[individuo].bvivo = false;
                };
            }

        }
    }
};

// inicia os pesos das redes neurais via gpu
__global__ void iniciarPesos(RedeNeural* d_individuos, int n, curandState* state){
    int individuo = blockIdx.x;     // 0–999  || warps
    int lane      = threadIdx.x;    // 0–31   || threads

    // não deixa passar do limite de warps
    if (individuo >= n) return;

    // só o thread 0 faz o trampo para o warp agir como unidade.
    if (lane == 0) {
        d_individuos[individuo].iniciarPesosDevice(d_individuos[individuo].rede, state);
    }
};

__global__ void mutarPesosDevice(RedeNeural* d_individuos, int n, curandState* state){
    int individuo = blockIdx.x;     // 0–999  || warps
    int lane      = threadIdx.x;    // 0–31   || threads

    // não deixa passar do limite de warps
    if (individuo >= n) return;

    // só o thread 0 faz o trampo para o warp agir como unidade.
    if (lane == 0) {
        d_individuos[individuo].mutacaoDevice(0.1, state);
    }
};

int main(){

    size_t sizeCandle = (sizeof(Candle) * 964800);
    size_t sizeIndividuos = (sizeof(RedeNeural) * NUM_INDIVIDUOS);
    //float taxaDeVitoriaDoMelhorIndividuo = 0;
    Melhores* h_melhores10 = new Melhores[10];
    Candle* h_candles = (Candle*)malloc(sizeCandle);
    RedeNeural* h_individuos = new RedeNeural[NUM_INDIVIDUOS];
    lerCSV_mallocc("./filtrado.csv", h_candles, 964800);

    // manda os candles para memoria da gpu
    Candle* d_candles;
    cudaMalloc(&d_candles, sizeCandle);
    cudaMemcpy(d_candles, h_candles, sizeCandle, cudaMemcpyHostToDevice);

    // manda os individuos para a memoria da gpu
    RedeNeural* d_individuos;
    cudaMalloc(&d_individuos, sizeIndividuos);
    cudaMemcpy(d_individuos, h_individuos, sizeIndividuos, cudaMemcpyHostToDevice);

    dim3 bloco(32);
    dim3 grid(NUM_INDIVIDUOS);

    // inicia o curandstate
    curandState* d_estados;
    int totalThreads = NUM_INDIVIDUOS * 32;
    cudaMalloc(&d_estados, totalThreads * sizeof(curandState));
    initCurand<<<grid, bloco>>>(d_estados, time(NULL));
    cudaDeviceSynchronize();

    // inicia os pesos dos individuos
    iniciarPesos<<<grid, bloco>>>(d_individuos, NUM_INDIVIDUOS, d_estados);
    cudaDeviceSynchronize();

    //* -----------------------LOOP PRINCIPAL DE TREINAMENTO---------------------------
    bool bmetaAtingida = false;
    while (bmetaAtingida == false) {
    
        // inicia treinamento(inferencia)
        treinamento<<<grid, bloco>>>(d_candles, d_individuos, NUM_INDIVIDUOS);
        cudaDeviceSynchronize();

        cudaMemcpy(h_individuos, d_individuos, sizeIndividuos, cudaMemcpyDeviceToHost);

        // cpu verifica se o individuo tem uma taxa de vitória maior que o melhor individuo histórico, se sim adiciona na lista dos melhores.
        for (size_t i = 0; i < NUM_INDIVIDUOS; i++) {
            if (h_individuos[i].bvivo == false) {
            continue;
            }

            int total = h_individuos[i].ganho = h_individuos[i].perda;
            float taxa = (((float)h_individuos[i].ganho / total) * 100);

            if (taxa >=h_melhores10[9].taxa) {

                // shift
                for (size_t ii = 0; ii < 9; ii++) {
                    h_melhores10[ii].rede = h_melhores10[ii+1].rede;
                    h_melhores10[ii].taxa = h_melhores10[ii+1].taxa;
                }

                h_melhores10[9].rede = h_individuos[i];
                h_melhores10[9].taxa = taxa;
            }
            
        }



        // TODO cpu repopula h_individuos com os 10 melhores, 10 aleatorios e o resto com copias dos 10 melhores.
        // TODO cpu manda os individuos para a memoria de gpu
        // TODO cpu chama kernel de mutar pesos
        
        // std::cout << "taxa de vitoria do melhor individuo: " << h_melhores10[9].taxa;
        // std::cout << std::endl;
        
        
        
        
        
        if (h_melhores10[9].taxa >=META_TAXA_VITORIA) {
            bmetaAtingida = true;
        }
            // TODO gpu começa uma nova rodada de inferencia
    }


    free(h_candles);
    delete[] h_individuos;
    delete[] h_melhores10;
    cudaFree(d_candles);
    cudaFree(d_individuos);
    cudaFree(d_estados);

    return 0;
}