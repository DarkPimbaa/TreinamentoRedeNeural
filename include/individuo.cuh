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

    __host__ __device__
    void init() {
        ganho = 0;
        perda = 0;
        taxaVitoria = 0.f;
        saidas = 0;
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
        // Calculate bias index taking broadcasting into account.
        // We assume 'dados' layout is [Individuo][MatrixCols].
        // Bias is [Individuo][Col].
        // Actually, Bias is [Individuo][Neuron].
        // Data is [Individuo][Neuron + BatchOffset].
        // The structure of Data depends on the Layer.
        // Layer < Last: NUM_NEURONIOS per column.
        // Layer == Last: NUM_SAIDAS per column.
        
        // Infer dimensions from size? No, that's messy.
        // But we know this function is called either for Hidden (NUM_NEURONIOS) or Output (NUM_SAIDAS).
        // The logic "idx % NUM_NEURONIOS" works if we pass the stride.
        // But we don't pass the stride.
        // HACK: Detect stride based on size? No, unreliable.
        // Let's assume standard behavior:
        // We effectively need to mod by the 'Height' of the matrix column.
        // If data is (NUM_INDIVIDUOS * NUM_NEURONIOS * BATCH), H = NUM_NEURONIOS.
        // If data is (NUM_INDIVIDUOS * NUM_SAIDAS * BATCH), H = NUM_SAIDAS.
        
        // Just relying on user to allow us to hardcode?
        // Let's assume NUM_NEURONIOS for now, but Output layer breaks this.
        
        // IMPORTANT: The original code used a flat AddBiasAndRelu.
        // Original: Bias size was [NUM_INDIVIDUOS * OutputSize].
        // Data size: [NUM_INDIVIDUOS * OutputSize].
        // So 1-to-1 mapping.
        // NOW: Data size > Bias size (because Batch > 1).
        // Bias is repeated for every Candle in the Batch.
        // Data: [Ind 0 [Candle 0][Candle 1]][Ind 1]...
        // Bias: [Ind 0             ][Ind 1]...
        
        // We need separate Kernels or pass the "Stride/Height" to this kernel.
        // I will change the signature.
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

        // Encoding output logic into a bitfield or temporary storage could be better, 
        // but 'Individuo' struct has only one 'saidas'. 
        // WARNING: storing 'saidas' implies we process them one by one or store history.
        // The original code only stored ONE action per individual (the latest one?). 
        // Wait, the original code ran LOOP over candles.
        // inside loop: inferencia -> check -> store action in 'saidas' -> verify -> update ganho/perda.
        // So 'saidas' is overwritten every candle.
        // We need to keep this behavior. 'saidas' is ephemeral.
        // Actually, we can just skip storing to 'saidas' if verify is done in same batch?
        // Let's modify verificarCompraVenda to take the raw output values directly!
        // So we don't need ReLuSaida to write to d_individuos.saidas if we verify immediately.
        // However, keeping modularity is good. 
        // Let's assume verifying is done in a separate kernel that reads memory.
        
        // Since 'saidas' is uint8_t, we clearly can't store 128 results there.
        // We MUST verify directly or use a temporary buffer.
        // Optimizing: Merge 'ReLuSaida' logic into 'verificarCompraVenda'.
    }
}

__global__ void verificarCompraVenda(Individuo *d_individuos, IndividuosPesos* d_individuosPesos, Candle *d_candles, int startCandle, int numCandles){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Indivíduo
    int candleBatchIdx = blockIdx.y; // Índice no Batch (0 a numCandles-1)

    if (idx < NUM_INDIVIDUOS && candleBatchIdx < numCandles) {
        int candleAtual = startCandle + candleBatchIdx;
        
        // Load outputs directly
        long long offset = (long long)idx * NUM_SAIDAS * CANDLE_BATCH_SIZE + (candleBatchIdx * NUM_SAIDAS);
        __half valVenda = d_individuosPesos->valoresOut[offset];
        __half valCompra = d_individuosPesos->valoresOut[offset + 1];
        
        // Converte para float para comparação
        float fValVenda = __half2float(valVenda);
        float fValCompra = __half2float(valCompra);
        
        // Solução 1: Lógica baseada em comparação de maior valor
        const float threshold = 0.01f;    // Confiança mínima para agir (reduzido)
        const float difference = 0.001f;  // Diferença mínima entre saídas (reduzido)
        
        int acao = 0;
        if (fValCompra > fValVenda + difference && fValCompra > threshold) {
            acao = 1; // Compra
        } else if (fValVenda > fValCompra + difference && fValVenda > threshold) {
            acao = 2; // Venda
        }

        float CandleAtualFechamento = d_candles[candleAtual].fechamento;
        float CandleDaFrenteFechamento = d_candles[candleAtual+1].fechamento;

        if (acao == 1) {
            //compra
            if (CandleAtualFechamento < CandleDaFrenteFechamento) {
                d_individuos[idx].ganho++;
            }else {
                d_individuos[idx].perda++;
            }

            if (CandleAtualFechamento == CandleDaFrenteFechamento) {
                d_individuos[idx].perda++;
            }
        } else if (acao == 2) {
            //venda
            if (CandleAtualFechamento > CandleDaFrenteFechamento) {
                d_individuos[idx].ganho++;
            }else {
                d_individuos[idx].perda++;
            }
            if (CandleAtualFechamento == CandleDaFrenteFechamento) {
                d_individuos[idx].perda++;
            }
        }
    }
};

__global__ void verificarMelhor(Individuo *d_individuos, int *d_melhor, int numCandlesBatch){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        int pontuacaoMelhor = INT_MIN;
        int idxMelhorQualificado = -1;
        int idxMelhorFallback = 0;  // Fallback para caso ninguém qualifique
        float melhorTaxaFallback = 0.0f;
        
        for (size_t i = 0; i < NUM_INDIVIDUOS; i++) {
            int totalTrades = d_individuos[i].ganho + d_individuos[i].perda;
            float taxaAcerto = (totalTrades > 0) 
                ? ((d_individuos[i].ganho / (float)totalTrades) * 100.0f) 
                : 0.0f;
            
            d_individuos[i].taxaVitoria = taxaAcerto;
            
            int pontuacao;
            
            // === NOVA LÓGICA: Taxa mínima de 60% ===
            if (taxaAcerto >= 60.0f && totalTrades > 0) {
                // Indivíduo qualificado: priorizar volume
                float bonusQualidade = (taxaAcerto - 60.0f) * 100.0f;
                float bonusVolume = (float)totalTrades * 10.0f;
                pontuacao = (int)(bonusQualidade + bonusVolume);
                
                if (pontuacao > pontuacaoMelhor) {
                    pontuacaoMelhor = pontuacao;
                    idxMelhorQualificado = i;
                }
            } else {
                // Fallback: guardar o melhor entre os não-qualificados
                if (taxaAcerto > melhorTaxaFallback || 
                    (taxaAcerto == melhorTaxaFallback && totalTrades > 
                     (d_individuos[idxMelhorFallback].ganho + d_individuos[idxMelhorFallback].perda))) {
                    melhorTaxaFallback = taxaAcerto;
                    idxMelhorFallback = i;
                }
            }
        }
        
        // Escolher melhor: qualificado ou fallback
        if (idxMelhorQualificado >= 0) {
            *d_melhor = idxMelhorQualificado;
        } else {
            *d_melhor = idxMelhorFallback;
        }
        
        int totalTradesMelhor = d_individuos[*d_melhor].ganho + d_individuos[*d_melhor].perda;
        float pctTrades = (float)totalTradesMelhor / (float)numCandlesBatch * 100.0f;
        
        printf("Taxa de vitoria do melhor individuo: %f\n", d_individuos[*d_melhor].taxaVitoria);
        printf("Ganho do melhor individuo: %i\n", d_individuos[*d_melhor].ganho);
        printf("Perda do melhor individuo: %i\n", d_individuos[*d_melhor].perda);
        printf("Trades: %d / %d (%.2f%%)\n", totalTradesMelhor, numCandlesBatch, pctTrades);
        printf("Qualificado (>= 60%%): %s\n", (idxMelhorQualificado >= 0) ? "SIM" : "NAO");
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
        
        // Caution: AddBiasAndRelu expects 'bias' to be broadcast?
        // Current 'AddBiasAndRelu': 
        // __global__ void AddBiasAndRelu(__half* dados, const __half* bias, int size)
        // bias size: [NUM_CAMADAS][NUM_INDIVIDUOS * NUM_NEURONIOS]
        // dados size: [NUM_INDIVIDUOS * NUM_NEURONIOS * CANDLE_BATCH_SIZE]
        // Bias needs to be repeated for each Candle in the Batch?
        // Bias depends on Neuron Index.
        // d_bias structure: [Individuo 0][Neuron 0..N], [Individuo 1]...
        // Data structure:   [Individuo 0][Neuron 0..N for Candle 0][Neuron 0..N for Candle 1]...
        // Wait.
        // Matrix C output: (m x n) = (NUM_NEURONIOS x CANDLE_BATCH_SIZE).
        // Stored Column Major.
        // [Neuron 0..N (C0)], [Neuron 0..N (C1)] ...
        // So for Individuo I:
        // Data block is size (NUM_NEURONIOS * CANDLE_BATCH_SIZE).
        // Bias block is size (NUM_NEURONIOS).
        // We need a modulated index for Bias.
        // AddBiasAndRelu is simple linear kernel.
        // We need to modify it or create a new one.
        // It's easier to make a Batched version.
        
        // Reusing AddBiasAndRelu logic with a hack?
        // No, let's just make a new simple kernel inline for now or assume modification.
        // Actually, I'll modify AddBiasAndRelu above.
        // Wait, I can't modify AddBiasAndRelu easily without changing its signature everywhere.
        // But it's only called here.
        // I will update AddBiasAndRelu logic in a separate edit if needed.
        // Actually, let's redefine AddBiasAndRelu to be aware of broadcasting.
        // But wait, the standard AddBiasAndRelu takes 'bias' array.
        // index 'idx' in data.
        // We need 'biasIdx'.
        // Data: [Ind 0 - Cand 0 - N 0..607] [Ind 0 - Cand 1 - N 0..607] ...
        // Bias: [Ind 0 - N 0..607]
        // biasIdx = (idx / (NUM_NEURONIOS * CANDLE_BATCH_SIZE)) * NUM_NEURONIOS + (idx % NUM_NEURONIOS).
        // Incorrect.
        // Data Layout: StrideC = NUM_NEURONIOS * CANDLE_BATCH_SIZE.
        // Individual I starts at I * StrideC.
        // Inside Individual I:
        //   Col 0 (Candle 0): 0..NUM_NEURONIOS-1
        //   Col 1 (Candle 1): NUM_NEURONIOS..2*NUM_NEURONIOS-1
        // Bias for Ind I:
        //   0..NUM_NEURONIOS-1.
        // So within one individual, bias repeats every NUM_NEURONIOS.
        // So biasIdx = (IndividuoIdx * NUM_NEURONIOS) + (NeuronIdx)
        // NeuronIdx = (idx % (NUM_NEURONIOS * CANDLE_BATCH_SIZE)) % NUM_NEURONIOS
        //           = idx % NUM_NEURONIOS.
        // IndividuoIdx = idx / (NUM_NEURONIOS * CANDLE_BATCH_SIZE).
        // So biasIdx = (idx / (NUM_NEURONIOS * CANDLE_BATCH_SIZE)) * NUM_NEURONIOS + (idx % NUM_NEURONIOS).
        // Yes.
        
        // I will update AddBiasAndRelu separately. For now, calling it with the right size but logic will be wrong if not updated.
        // I will perform the update to AddBiasAndRelu in another chunk.
    
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
        // Wait. Last layer weights:
        // Struct: __half pesos[NUM_CAMADAS][NUM_INDIVIDUOS * NUM_NEURONIOS * NUM_NEURONIOS];
        // Allocated as NUM_NEURONIOS squared. But Last layer uses (NUM_SAIDAS x NUM_NEURONIOS)?
        // If so, 2x608.
        // If the array is packed 608x608, we can just use the first part.
        // But strideA MUST match the stored gap between Individual Matrices.
        // The weight array has stride `NUM_NEURONIOS * NUM_NEURONIOS`.
        // So even if we use smaller matrix, the stride is the full size. Correct.
        
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha, d_A, CUDA_R_16F, m, NUM_NEURONIOS * NUM_NEURONIOS,
            d_B, CUDA_R_16F, k, (long long)NUM_NEURONIOS * CANDLE_BATCH_SIZE, // This strideB was wrong in original code?? 
                                              // "d_B, CUDA_R_16F, k, NUM_NEURONIOS" -> 4th param is ldb. 
                                              // 5th param is strideB.
            // Original code:
            // d_B, CUDA_R_16F, k, NUM_NEURONIOS,
            // strideB = NUM_NEURONIOS. 
            // This assumes B is [NUM_INDIVIDUOS][NUM_NEURONIOS].
            // Correct.
            // Now B is [NUM_INDIVIDUOS][NUM_NEURONIOS * CANDLE_BATCH_SIZE].
            // So strideB = NUM_NEURONIOS * CANDLE_BATCH_SIZE.
            // ldb = k = NUM_NEURONIOS.
            
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
