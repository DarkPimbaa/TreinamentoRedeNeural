#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cstdio>
#include "./types.hpp"

#ifndef NUM_INDIVIDUOS
#define NUM_INDIVIDUOS 1000
#endif
#ifndef NUM_CAMADAS
#define NUM_CAMADAS 4
#endif
#ifndef NUM_NEURONIOS
#define NUM_NEURONIOS 608 // multiplo de 16
#endif
#ifndef NUM_SAIDAS
#define NUM_SAIDAS 2
#endif

#ifndef CANDLE_BATCH_SIZE
#define CANDLE_BATCH_SIZE 128
#endif

#ifndef MAXIMO_RODADAS_SEM_TRADES
#define MAXIMO_RODADAS_SEM_TRADES 120
#endif

// Constante de Bias solicitada
#define BIAS 1.0f

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) \
    { \
        cublasStatus_t err = call; \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            printf("Erro cuBLAS linha %d: %d\n", __LINE__, err); \
        } \
    }
#endif

struct Individuo {
    int ganho;
    int perda;
    float taxaVitoria;
    uint8_t saidas;
    curandState state; // Estado do gerador aleatório para cada individuo
    
    // Sistema Bvivo
    int rodadasSemTrade;  // Contador de inatividade
    bool bvivo;           // true = vivo, false = morto

    __host__ __device__
    void init() {
        ganho = 0;
        perda = 0;
        taxaVitoria = 0.f;
        saidas = 0;
        rodadasSemTrade = 0;
        bvivo = true;
    }
};

struct IndividuosPesos {
    // Pesos: [Camada][Individuo * Input * Output]
    __half pesos[NUM_CAMADAS][NUM_INDIVIDUOS * NUM_NEURONIOS * NUM_NEURONIOS];
    
    // Bias: [Camada][Individuo * Output] - Cada neurônio tem um peso de bias
    // Alocamos tamanho máximo (NUM_NEURONIOS) para simplificar, a última camada usa menos.
    __half bias[NUM_CAMADAS][NUM_INDIVIDUOS * NUM_NEURONIOS];

    __half valoresNormalizados[NUM_NEURONIOS * CANDLE_BATCH_SIZE]; 
    __half valoresMid[NUM_INDIVIDUOS * NUM_NEURONIOS * CANDLE_BATCH_SIZE]; 
    __half valoresSwap[NUM_INDIVIDUOS * NUM_NEURONIOS * CANDLE_BATCH_SIZE]; 
    __half valoresOut[NUM_INDIVIDUOS * NUM_SAIDAS * CANDLE_BATCH_SIZE]; 
};

// --- KERNELS DE INFERÊNCIA ---

// Aplica Bias + ReLu: Output = Max(0, Input + Bias * 1.0)
__global__ void AddBiasAndRelu(__half* dados, const __half* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
    }
}
// New Signature implementation
__global__ void AddBiasAndRelu(__half* dados, const __half* bias, int size, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int batchChunkSize = height * CANDLE_BATCH_SIZE;
        int individuoIdx = idx / batchChunkSize;
        int rowIdx = idx % height; 
        
        int biasIdx = individuoIdx * height + rowIdx;

        float val = __half2float(dados[idx]);
        float valBias = __half2float(bias[biasIdx]);
        
        val += valBias;

        if (val < 0.0f) {
            val = 0.0f;
        }
        dados[idx] = __float2half(val);
    }
}

__global__ void ReLuSaida(Individuo* d_individuos, IndividuosPesos* d_individuosPesos, int numCandles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Individuo Index
    int candleIdx = blockIdx.y; // Candle Index within batch

    if (idx < NUM_INDIVIDUOS && candleIdx < numCandles) {
        // Output 0: Venda | Output 1: Compra
        
        long long offset = (long long)idx * NUM_SAIDAS * CANDLE_BATCH_SIZE + (candleIdx * NUM_SAIDAS);
        
        __half valVenda = d_individuosPesos->valoresOut[offset];
        __half valCompra = d_individuosPesos->valoresOut[offset + 1];
        __half zero = __float2half(0.f);
    }
}

__global__ void verificarCompraVenda(Individuo *d_individuos, IndividuosPesos* d_individuosPesos, Candle *d_candles, int startCandle, int numCandles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indivíduo
    int candleBatchIdx = blockIdx.y; // Índice no Batch (0 a numCandles-1)

    if (idx < NUM_INDIVIDUOS && candleBatchIdx < numCandles) {
        // Skip se já está morto
        if (!d_individuos[idx].bvivo) return;
        
        int candleAtual = startCandle + candleBatchIdx;
        
        // Load outputs directly
        long long offset = (long long)idx * NUM_SAIDAS * CANDLE_BATCH_SIZE + (candleBatchIdx * NUM_SAIDAS);
        __half valVenda = d_individuosPesos->valoresOut[offset];
        __half valCompra = d_individuosPesos->valoresOut[offset + 1];
        
        // Converte para float para comparação
        float fValOut1 = __half2float(valVenda);
        float fValOut2 = __half2float(valCompra);
        
        int acao = 0;
        // Lógica solicitada:
        // Se neuronio 1 for 0 e neuronio 2 for true (>0) -> Venda
        // Se neuronio 1 for true (>0) e neuronio 2 for 0 -> Compra
        // Caso ambos false (0) ou ambos true (>0) -> Nada
        
        if (fValOut1 == 0.0f && fValOut2 > 0.0f) {
            acao = 2; // Venda
        } else if (fValOut1 > 0.0f && fValOut2 == 0.0f) {
            acao = 1; // Compra
        }

        float CandleAtualFechamento = d_candles[candleAtual].fechamento;
        float CandleDaFrenteFechamento = d_candles[candleAtual+1].fechamento;

        if (acao == 1) {
            //compra
            if (CandleAtualFechamento < CandleDaFrenteFechamento) {
                atomicAdd(&d_individuos[idx].ganho, 1);
            } else if (CandleAtualFechamento > CandleDaFrenteFechamento) {
                // Queda = perda (empate não conta)
                atomicAdd(&d_individuos[idx].perda, 1);
            }
            // Se == não faz nada (empate neutro)
            
            // Fez trade: reseta contador de inatividade
            d_individuos[idx].rodadasSemTrade = 0;
        } else if (acao == 2) {
            //venda
            if (CandleAtualFechamento > CandleDaFrenteFechamento) {
                atomicAdd(&d_individuos[idx].ganho, 1);
            } else if (CandleAtualFechamento < CandleDaFrenteFechamento) {
                // Subida = perda (empate não conta)
                atomicAdd(&d_individuos[idx].perda, 1);
            }
            // Se == não faz nada (empate neutro)
            
            // Fez trade: reseta contador de inatividade
            d_individuos[idx].rodadasSemTrade = 0;
        } else {
            // Não fez nada: incrementa contador e verifica morte
            int novoValor = atomicAdd(&d_individuos[idx].rodadasSemTrade, 1) + 1;
            if (novoValor >= MAXIMO_RODADAS_SEM_TRADES) {
                d_individuos[idx].bvivo = false;
            }
        }
    }
};

__global__ void verificarMelhor(Individuo *d_individuos, int *d_melhor, int numCandlesBatch){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int idxMelhorVivo = -1;
        float melhorTaxaVivo = -1.0f;
        
        int idxMelhorMorto = 0;  // Fallback: melhor entre os mortos
        float melhorTaxaMorto = -1.0f;
        
        int vivosCount = 0;
        
        for (size_t i = 0; i < NUM_INDIVIDUOS; i++) {
            int totalTrades = d_individuos[i].ganho + d_individuos[i].perda;
            float taxaAcerto = (totalTrades > 0) 
                ? ((d_individuos[i].ganho / (float)totalTrades) * 100.0f) 
                : 0.0f;
            
            d_individuos[i].taxaVitoria = taxaAcerto;
            
            if (d_individuos[i].bvivo) {
                // Indivíduo VIVO: candidato válido
                vivosCount++;
                
                // Prioriza APENAS taxa de acerto
                if (taxaAcerto > melhorTaxaVivo) {
                    melhorTaxaVivo = taxaAcerto;
                    idxMelhorVivo = i;
                }
            } else {
                // Indivíduo MORTO: guardar para fallback
                if (taxaAcerto > melhorTaxaMorto) {
                    melhorTaxaMorto = taxaAcerto;
                    idxMelhorMorto = i;
                }
            }
        }
        
        // Escolher melhor: vivo ou ressuscitar morto
        if (idxMelhorVivo >= 0) {
            *d_melhor = idxMelhorVivo;
        } else {
            // TODOS MORRERAM: ressuscitar o melhor cadáver
            *d_melhor = idxMelhorMorto;
            d_individuos[idxMelhorMorto].bvivo = true;
            d_individuos[idxMelhorMorto].rodadasSemTrade = 0;
            printf("!!! TODOS MORRERAM - Ressuscitando individuo %d !!!\n", idxMelhorMorto);
        }
        
        int totalTradesMelhor = d_individuos[*d_melhor].ganho + d_individuos[*d_melhor].perda;
        float pctTrades = (float)totalTradesMelhor / (float)numCandlesBatch * 100.0f;
        
        printf("Vivos: %d / %d\n", vivosCount, NUM_INDIVIDUOS);
        printf("==============================================\n");
        printf("Taxa de vitoria do melhor: %.2f%%\n", d_individuos[*d_melhor].taxaVitoria);
        printf("==============================================\n");
        printf("Ganho: %d | Perda: %d\n", d_individuos[*d_melhor].ganho, d_individuos[*d_melhor].perda);
        printf("==============================================\n");
        printf("Trades: %d / %d (%.2f%%)\n", totalTradesMelhor, numCandlesBatch, pctTrades);
        printf("==============================================\n");
    }
};

__global__ void zeraIndividuos(Individuo *d_individuos){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_INDIVIDUOS) {
        d_individuos[idx].init();
    }
}

/**
* normaliza os valores pelo fechamento do candle atual
* os ultimos 8 indices são os suportes e resistencias
*/
__global__ void normalizarValores(IndividuosPesos *d_pesos, Candle *d_candles, int startCandle, int numCandles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Neurônio
    int candleBatchIdx = blockIdx.y; // Índice do Candle no Batch

    if (idx < NUM_NEURONIOS && candleBatchIdx < numCandles) {
        int candleAtual = startCandle + candleBatchIdx;
        
        // Fatores de normalização baseados no candle atual
        float factor = d_candles[candleAtual].fechamento;
        float factorTrades = d_candles[candleAtual].trades; 
        float factorVolume = d_candles[candleAtual].volume;

        // Evita divisão por zero
        if (factor == 0.0f) factor = 1.0f;
        if (factorTrades == 0.0f) factorTrades = 1.0f;
        if (factorVolume == 0.0f) factorVolume = 1.0f;

        float val = 0.0f;

        // --- Inputs Históricos (Índices 0 a 599) ---
        // 100 candles * 6 atributos
        if (idx < 600) {
            int inputsPorCandle = 6;
            int candleOffset = idx / inputsPorCandle; // 0 a 99 (offset relativo)
            int atributo = idx % inputsPorCandle;     // 0 a 5

            // Recupera o candle correspondente no histórico (Cronológico: T-99 até T)
            int targetCandleIdx = candleAtual - 99 + candleOffset;
            
            // Leitura direta da memória global
            Candle c = d_candles[targetCandleIdx];

            switch(atributo) {
                case 0: val = c.abertura / factor; break;
                case 1: val = c.maxima / factor; break;
                case 2: val = c.minima / factor; break;
                case 3: val = c.fechamento / factor; break;
                case 4: val = c.volume / factorVolume; break;
                case 5: val = c.trades / factorTrades; break;
            }
        }
        // --- Inputs Suportes e Resistências (Índices 600 a 607) ---
        else {
            Candle cAtual = d_candles[candleAtual];
            int srIdx = idx - 600;
            switch(srIdx) {
                case 0: val = cAtual.s15; break;
                case 1: val = cAtual.r15; break;
                case 2: val = cAtual.s30; break;
                case 3: val = cAtual.r30; break;
                case 4: val = cAtual.s60; break;
                case 5: val = cAtual.r60; break;
                case 6: val = cAtual.s180; break;
                case 7: val = cAtual.r180; break;
            }
            val /= factor;
        }

        // Armazena no buffer (Column-Major layout: todas as inputs do candle 0, depois candle 1...)
        // B matrix is (NUM_NEURONIOS x CANDLE_BATCH_SIZE)
        // Element at (row=idx, col=candleBatchIdx)
        // Index = row + col * lda = idx + candleBatchIdx * NUM_NEURONIOS
        d_pesos->valoresNormalizados[idx + candleBatchIdx * NUM_NEURONIOS] = __float2half(val);
    }
}

__host__ void inferencia(Individuo* d_individuos, IndividuosPesos* d_individuosPesos, int numCamadas, int numCandles) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    }

    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // --- Loop das Camadas Ocultas ---
    for (int i = 0; i < numCamadas - 1; ++i) {
        int m = NUM_NEURONIOS; 
        int n = numCandles;             
        int k = NUM_NEURONIOS; 

        const __half* d_A = d_individuosPesos->pesos[i];
        const __half* d_B = (i == 0) ? d_individuosPesos->valoresNormalizados : d_individuosPesos->valoresMid;
        __half* d_C = d_individuosPesos->valoresSwap;

        long long int strideA = NUM_NEURONIOS * NUM_NEURONIOS;
        
        // StrideB: Distance between matrices.
        // For Layer 0: B is SHARED (1 matrix for all batches). Stride = 0.
        // For Layer >0: B is UNIQUE (1 matrix per batch). Stride = NUM_NEURONIOS * CANDLE_BATCH_SIZE (or numCandles? Buffer is fixed size).
        // Buffer is allocated for CANDLE_BATCH_SIZE.
        long long int strideB = (i == 0) ? 0 : (long long)NUM_NEURONIOS * CANDLE_BATCH_SIZE;
        long long int strideC = (long long)NUM_NEURONIOS * CANDLE_BATCH_SIZE;

        // 1. Multiplicação de Matriz (Pesos * Input)
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha, d_A, CUDA_R_16F, m, strideA,
            d_B, CUDA_R_16F, k, strideB,
            &beta, d_C, CUDA_R_16F, m, strideC,
            NUM_INDIVIDUOS, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // 2. Soma Bias e Aplica ReLu
        int totalElements = NUM_INDIVIDUOS * NUM_NEURONIOS * CANDLE_BATCH_SIZE;
        int threads = 256;
        int blocks = (totalElements + threads - 1) / threads; // Naively big grid
    
        AddBiasAndRelu<<<blocks, threads>>>(d_individuosPesos->valoresSwap, d_individuosPesos->bias[i], totalElements);
        
        // Copia Swap -> Mid para a próxima camada
        cudaMemcpy(d_individuosPesos->valoresMid, d_individuosPesos->valoresSwap, totalElements * sizeof(__half), cudaMemcpyDeviceToDevice);
    }

    // --- Última Camada (Output) ---
    {
        int m = NUM_SAIDAS;
        int n = numCandles;
        int k = NUM_NEURONIOS;

        const __half* d_A = d_individuosPesos->pesos[numCamadas - 1];
        const __half* d_B = d_individuosPesos->valoresMid;
        __half* d_C = d_individuosPesos->valoresOut;

        long long int strideA = NUM_NEURONIOS * NUM_NEURONIOS; // This is wrong. Last layer is (NUM_SAIDAS x NUM_NEURONIOS)?
       
        
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha, d_A, CUDA_R_16F, m, NUM_NEURONIOS * NUM_NEURONIOS,
            d_B, CUDA_R_16F, k, (long long)NUM_NEURONIOS * CANDLE_BATCH_SIZE, // This strideB was wrong in original code?? 
     
            
            &beta, d_C, CUDA_R_16F, m, (long long)m * CANDLE_BATCH_SIZE, // strideC = m * CANDLE_BATCH_SIZE
            // ldc = m.
            NUM_INDIVIDUOS, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Bias + ReLu na Saída (IMPORTANTE: Bias da última camada)
        int totalElements = NUM_INDIVIDUOS * NUM_SAIDAS * CANDLE_BATCH_SIZE;
        int threads = 128;
        int blocks = (totalElements + threads - 1) / threads;
        AddBiasAndRelu<<<blocks, threads>>>(d_individuosPesos->valoresOut, d_individuosPesos->bias[numCamadas - 1], totalElements);
    }
};


// --- KERNELS DE MUTAÇÃO (DINOSAUR LOGIC) ---
/**
 * @brief Kernel de Clonagem em Massa (Broadcasting DNA).
 * Pega o material genético do melhor indivíduo (idx_origem) e replica para 
 * todos os outros indivíduos da população.
 */
__global__ void ClonarDNAMassa(IndividuosPesos* d_pesos, int *idx_origem) {
    // Cada thread foca em um "gene" (um peso ou um bias específico)
    int geneIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // --- Clonagem de Pesos ---
    // Total de pesos por matriz: NUM_NEURONIOS * NUM_NEURONIOS
    if (geneIdx < NUM_NEURONIOS * NUM_NEURONIOS) {
        for (int c = 0; c < NUM_CAMADAS; c++) {
            // long long layerOffset = c * (long long)(NUM_INDIVIDUOS * NUM_NEURONIOS * NUM_NEURONIOS); // REMOVIDO
            long long sizeMatriz = (long long)NUM_NEURONIOS * NUM_NEURONIOS;
            
            // Valor do DNA do "Mestre"
            // layerOffset removido aqui pois d_pesos->pesos[c] já aponta para a camada correta
            __half dnaMestre = d_pesos->pesos[c][(*idx_origem * sizeMatriz) + geneIdx];

            // Replica para toda a população
            for (int d = 0; d < NUM_INDIVIDUOS; d++) {
                if (d == *idx_origem) continue; // Não sobrescreve a si mesmo
                
                long long posDestino = (d * sizeMatriz) + geneIdx;
                d_pesos->pesos[c][posDestino] = dnaMestre;
            }
        }
    }
    
    // --- Clonagem de Bias ---
    // Usamos as mesmas threads, mas limitamos ao range de neurônios
    if (geneIdx < NUM_NEURONIOS) {
        for (int c = 0; c < NUM_CAMADAS; c++) {
            // long long layerOffset = c * (long long)(NUM_INDIVIDUOS * NUM_NEURONIOS); // REMOVIDO
            long long sizeBias = (long long)NUM_NEURONIOS;
            
            __half biasMestre = d_pesos->bias[c][(*idx_origem * sizeBias) + geneIdx];

            for (int d = 0; d < NUM_INDIVIDUOS; d++) {
                if (d == *idx_origem) continue;
                
                long long posDestino = (d * sizeBias) + geneIdx;
                d_pesos->bias[c][posDestino] = biasMestre;
            }
        }
    }
}
/**
 * @brief Kernel de Mutação Global.
 * Aplica mutações em todos os indivíduos da população, exceto no campeão (idx_preservado).
 * @param d_individuos Vetor para acessar o curandState de cada um.
 * @param d_pesos Estrutura de pesos e biases.
 * @param idx_preservado ID do indivíduo que NÃO deve sofrer mutação (o melhor da geração).
 * @param rangeRandom Intensidade/quantidade de mutações.
 */
__global__ void AplicarMutacaoGlobal(Individuo* d_individuos, IndividuosPesos* d_pesos, int *idx_preservado, float rangeRandom) {
    int id_individuo = blockIdx.x * blockDim.x + threadIdx.x;

    // Garante que estamos no range e pula o campeão
    if (id_individuo < NUM_INDIVIDUOS && id_individuo != *idx_preservado) {
        curandState* state = &d_individuos[id_individuo].state;

        // Determina quantas mutações fazer neste indivíduo
        int mutations = (curand(state) % (int)rangeRandom) + 1;

        for (int k = 0; k < mutations; k++) {
            int tipo = curand(state) % 3;
            int camada = curand(state) % NUM_CAMADAS;
            
            // 90% chance de mutar peso, 10% bias
            bool mutarBias = (curand_uniform(state) > 0.9f);
            
            int maxIndex;
            __half* alvo;

            if (mutarBias) {
                maxIndex = NUM_NEURONIOS;
                long long offset = (long long)id_individuo * NUM_NEURONIOS;
                alvo = &d_pesos->bias[camada][offset];
            } else {
                maxIndex = NUM_NEURONIOS * NUM_NEURONIOS;
                long long offset = (long long)id_individuo * NUM_NEURONIOS * NUM_NEURONIOS;
                alvo = &d_pesos->pesos[camada][offset];
            }

            int indice = curand(state) % maxIndex;
            float valorAtual = __half2float(alvo[indice]);
            float novoValor = valorAtual;

            // Lógica de mutação "Dinosaur"
            if (tipo == 0) { // Reset aleatório
                novoValor = curand_uniform(state) * 2.0f - 1.0f; 
            }
            else if (tipo == 1) { // Multiplicação escalar
                float fator = curand_uniform(state) + 0.5f; // [0.5, 1.5]
                novoValor *= fator;
            }
            else if (tipo == 2) { // Soma de ruído
                float adicao = (curand_uniform(state) * 2.0f - 1.0f) / 100.0f; // [-0.01, 0.01]
                novoValor += adicao;
            }

            alvo[indice] = __float2half(novoValor);
        }
    }
}

// usado para faciliar no calculo de quantos blocos um kernel precisa
int getblocksize(int N, int threads){
   int blocks = (N + threads - 1) / threads;
   return blocks;
};

/**
 * @brief Inicializa o estado do gerador de números aleatórios (cuRAND) para cada indivíduo.
 * @param d_individuos Ponteiro para o array de indivíduos na VRAM.
 * @param seed Semente base (geralmente time(NULL) vindo do Host) para garantir aleatoriedade entre sessões.
 */
__global__ void IniciarEstadosCurand(Individuo* d_individuos, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < NUM_INDIVIDUOS) {
        // Cada thread recebe a mesma seed, mas uma sequência diferente baseada no seu ID.
        // O terceiro parâmetro (offset) é 0.
        curand_init(seed, id, 0, &d_individuos[id].state);
    }
};
/**
 * @brief Kernel para inicialização de pesos e biases no range estável [-1.0, 1.0].
 * Garante que a soma dos sinais não ultrapasse o limite de precisão do FP16 (65k).
 * @param d_individuos Vetor para acesso ao curandState.
 * @param d_pesos Estrutura de pesos e biases.
 */
__global__ void IniciarPesosEbias(Individuo* d_individuos, IndividuosPesos* d_pesos) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < NUM_INDIVIDUOS) {
        curandState* state = &d_individuos[id].state;
        
        // Range: [-1.0, 1.0]
        float range = 2.0f;
        float offset = -1.0f;

        // Inicializa Pesos
        for (int c = 0; c < NUM_CAMADAS; c++) {
            long long startIdx = (long long)id * NUM_NEURONIOS * NUM_NEURONIOS;
            for (int i = 0; i < NUM_NEURONIOS * NUM_NEURONIOS; i++) {
                float r = curand_uniform(state) * range + offset;
                d_pesos->pesos[c][startIdx + i] = __float2half(r);
            }
        }

        // Inicializa Bias
        for (int c = 0; c < NUM_CAMADAS; c++) {
            long long startIdxBias = (long long)id * NUM_NEURONIOS;
            for (int i = 0; i < NUM_NEURONIOS; i++) {
                float r = curand_uniform(state) * range + offset;
                d_pesos->bias[c][startIdxBias + i] = __float2half(r);
            }
        }
    }
};
