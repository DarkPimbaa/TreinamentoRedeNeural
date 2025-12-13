#include <cstdio>
#define NUM_ENTRADAS 60 // 6 valores por candle * 100 candles
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2
#define BIAS 1.f
#define NUM_INDIVIDUOS 1000 // REDUZIDO! Com 600 entradas cada rede usa ~3.6MB
#define NUM_ELITE 10       // quantidade de melhores preservados
#define NUM_ALEATORIOS                                                         \
  100 // quantidade de indivíduos aleatórios para diversidade
#define INICIO_MUTACOES                                                        \
  (NUM_ELITE + NUM_ALEATORIOS) // índice onde começam as mutações
#define META_TAXA_VITORIA 60

#include "includes/redeNeural.cu"
#include "includes/types.hpp"
#include "includes/utils.hpp"
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <time.h>

struct Melhores {
  RedeNeural rede;
  float taxa = 0;
  bool bValido = false; // indica se este slot contém um melhor válido
};

__global__ void initCurand(curandState *estados, unsigned long seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &estados[tid]);
}

__global__ void treinamento(Candle *d_candles, RedeNeural *d_individuos,
                            int n) {
  int individuo = blockIdx.x;
  int lane = threadIdx.x;

  if (individuo >= n)
    return;

  if (lane == 0) {
    int janela = NUM_ENTRADAS / 6; // 100 candles quando NUM_ENTRADAS=600
    float valores[NUM_ENTRADAS];   // array dinâmico baseado em NUM_ENTRADAS

    // itera sobre todo o dataset
    for (size_t i = (janela - 1); i < 4000 /*964799*/; i++) {

      // O "factor" é a abertura do candle atual (o mais recente da janela)
      float factor = d_candles[i].abertura;

      // Proteção contra divisão por zero
      if (factor < 0.00001f)
        factor = 1.0f;

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
        const Candle &c = d_candles[indice_leitura];

        valores[idx_val + 0] = c.abertura / factor;
        valores[idx_val + 1] = c.maxima / factor;
        valores[idx_val + 2] = c.minima / factor;
        valores[idx_val + 3] = c.fechamento / factor;

        valores[idx_val + 4] = (float)c.volume;
        valores[idx_val + 5] = (float)c.trades;

        idx_val += 6;
      }

      // Alimenta a rede neural
      d_individuos[individuo].iniciar(valores);

      // verifica se ganhou ou perdeu
      if (d_individuos[individuo].retorno[0] == true &&
          d_individuos[individuo].retorno[1] == false) {
        // verifica se o proximo candle é de alta, se sim ganhou se não perdeu
        if (d_candles[i].fechamento < d_candles[i + 1].fechamento) {
          d_individuos[individuo].ganho++;
        } else {
          d_individuos[individuo].perda++;
        }

        d_individuos[individuo].rodadasSemApostar = 0;
      } else if (d_individuos[individuo].retorno[0] == false &&
                 d_individuos[individuo].retorno[1] == true) {
        // verifica se o proximo candle é de baixa, se sim ganhou se não perdeu
        if (d_candles[i].fechamento > d_candles[i + 1].fechamento) {
          d_individuos[individuo].ganho++;
        } else {
          d_individuos[individuo].perda++;
        }

        d_individuos[individuo].rodadasSemApostar = 0;
      } else {
        // não apostou
        d_individuos[individuo].rodadasSemApostar++;

        if (d_individuos[individuo].rodadasSemApostar >= 100) {
          d_individuos[individuo].bvivo = false;
        };
      }
    }
  }
};

// inicia os pesos das redes neurais via gpu
__global__ void iniciarPesos(RedeNeural *d_individuos, int n,
                             curandState *state) {
  int individuo = blockIdx.x; // 0–999  || warps
  int lane = threadIdx.x;     // 0–31   || threads

  // não deixa passar do limite de warps
  if (individuo >= n)
    return;

  // só o thread 0 faz o trampo para o warp agir como unidade.
  if (lane == 0) {
    d_individuos[individuo].iniciarPesosDevice(d_individuos[individuo].rede,
                                               state);
  }
};

// Kernel para iniciar pesos apenas dos indivíduos em um range específico
__global__ void iniciarPesosRange(RedeNeural *d_individuos, int startIdx,
                                  int endIdx, curandState *state) {
  int individuo = blockIdx.x + startIdx; // offset pelo startIdx
  int lane = threadIdx.x;

  // só processa indivíduos dentro do range
  if (individuo >= endIdx)
    return;

  // só o thread 0 faz o trabalho
  if (lane == 0) {
    d_individuos[individuo].iniciarPesosDevice(d_individuos[individuo].rede,
                                               state);
  }
};

__global__ void mutarPesosDevice(RedeNeural *d_individuos, int startIdx, int n,
                                 curandState *state) {
  int individuo = blockIdx.x; // 0–999  || warps
  int lane = threadIdx.x;     // 0–31   || threads

  // não deixa passar do limite de warps
  if (individuo >= n)
    return;

  // pula a elite e os aleatórios (só muta a partir de startIdx)
  if (individuo < startIdx)
    return;

  // só o thread 0 faz o trampo para o warp agir como unidade.
  if (lane == 0) {
    d_individuos[individuo].mutacaoDevice(0.05, state);
  }
};

int main() {

  size_t sizeCandle = (sizeof(Candle) * 964800);
  size_t sizeIndividuos = (sizeof(RedeNeural) * NUM_INDIVIDUOS);

  // Verificação de memória GPU
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  size_t requiredMem =
      sizeCandle + sizeIndividuos + (NUM_INDIVIDUOS * 32 * sizeof(curandState));

  std::cout << "=== Verificação de Memória ===" << std::endl;
  std::cout << "Tamanho de cada RedeNeural: "
            << (sizeof(RedeNeural) / 1024 / 1024) << " MB" << std::endl;
  std::cout << "Memória para indivíduos: " << (sizeIndividuos / 1024 / 1024)
            << " MB" << std::endl;
  std::cout << "Memória GPU disponível: " << (freeMem / 1024 / 1024) << " MB"
            << std::endl;
  std::cout << "Memória GPU necessária: " << (requiredMem / 1024 / 1024)
            << " MB" << std::endl;

  if (requiredMem > freeMem) {
    std::cerr << "ERRO: Memória GPU insuficiente! Reduza NUM_INDIVIDUOS."
              << std::endl;
    return 1;
  }
  std::cout << "==============================" << std::endl;

  Melhores *h_melhores10 = new Melhores[NUM_ELITE];
  Candle *h_candles = (Candle *)malloc(sizeCandle);
  RedeNeural *h_individuos = new RedeNeural[NUM_INDIVIDUOS];
  lerCSV_mallocc("./filtrado.csv", h_candles, 964800);

  // manda os candles para memoria da gpu
  Candle *d_candles;
  cudaMalloc(&d_candles, sizeCandle);
  cudaMemcpy(d_candles, h_candles, sizeCandle, cudaMemcpyHostToDevice);

  // manda os individuos para a memoria da gpu
  RedeNeural *d_individuos;
  cudaMalloc(&d_individuos, sizeIndividuos);
  cudaMemcpy(d_individuos, h_individuos, sizeIndividuos,
             cudaMemcpyHostToDevice);

  dim3 bloco(32);
  dim3 grid(NUM_INDIVIDUOS);

  // inicia o curandstate
  curandState *d_estados;
  int totalThreads = NUM_INDIVIDUOS * 32;
  cudaMalloc(&d_estados, totalThreads * sizeof(curandState));
  initCurand<<<grid, bloco>>>(d_estados, time(NULL));
  cudaDeviceSynchronize();

  // inicia os pesos dos individuos
  iniciarPesos<<<grid, bloco>>>(d_individuos, NUM_INDIVIDUOS, d_estados);
  cudaDeviceSynchronize();

  //* -----------------------LOOP PRINCIPAL DE
  // TREINAMENTO---------------------------
  bool bmetaAtingida = false;
  while (bmetaAtingida == false) {

    // inicia treinamento(inferencia)
    treinamento<<<grid, bloco>>>(d_candles, d_individuos, NUM_INDIVIDUOS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_individuos, d_individuos, sizeIndividuos,
               cudaMemcpyDeviceToHost);

    // cpu verifica se o individuo tem uma taxa de vitória maior que o melhor
    // individuo histórico, se sim adiciona na lista dos melhores.
    for (size_t i = 0; i < NUM_INDIVIDUOS; i++) {
      if (h_individuos[i].bvivo == false) {
        continue;
      }

      int total = h_individuos[i].ganho + h_individuos[i].perda;
      if (total == 0)
        continue; // evita divisão por zero

      float taxa = (((float)h_individuos[i].ganho / total) * 100);

      if (taxa >= h_melhores10[9].taxa) {
        // shift para baixo (índice 0 é descartado, 9 é o melhor)
        for (size_t ii = 0; ii < 9; ii++) {
          h_melhores10[ii].rede = h_melhores10[ii + 1].rede;
          h_melhores10[ii].taxa = h_melhores10[ii + 1].taxa;
          h_melhores10[ii].bValido = h_melhores10[ii + 1].bValido;
        }

        h_melhores10[9].rede = h_individuos[i];
        h_melhores10[9].taxa = taxa;
        h_melhores10[9].bValido = true; // marca como válido
      }
    }

    // Repopula h_individuos com estratégia evolutiva:
    // - Índices 0 a NUM_ELITE: os melhores históricos válidos (elite)
    // - Índices NUM_ELITE a INICIO_MUTACOES: indivíduos aleatórios
    // (diversidade)
    // - Índices INICIO_MUTACOES+: cópias dos melhores válidos (para mutação)

    // Conta quantos melhores válidos temos
    int numMelhoresValidos = 0;
    for (int i = 0; i < NUM_ELITE; i++) {
      if (h_melhores10[i].bValido)
        numMelhoresValidos++;
    }

    // 1. Copia os melhores válidos para as primeiras posições
    int idxIndividuo = 0;
    for (int i = 0; i < NUM_ELITE && idxIndividuo < NUM_ELITE; i++) {
      if (h_melhores10[i].bValido) {
        h_individuos[idxIndividuo] = h_melhores10[i].rede;
        idxIndividuo++;
      }
    }

    // 2. Preenche até INICIO_MUTACOES com indivíduos aleatórios
    // (inclui slots onde não havia melhores válidos)
    for (int i = idxIndividuo; i < INICIO_MUTACOES; i++) {
      h_individuos[i] = RedeNeural(); // reset para novo indivíduo (aleatório)
    }

    // 3. Preenche o resto com cópias dos melhores válidos (se houver)
    if (numMelhoresValidos > 0) {
      int idxMelhor = 0;
      for (int i = INICIO_MUTACOES; i < NUM_INDIVIDUOS; i++) {
        // Encontra o próximo melhor válido (circular)
        while (!h_melhores10[idxMelhor % NUM_ELITE].bValido) {
          idxMelhor++;
        }
        h_individuos[i] = h_melhores10[idxMelhor % NUM_ELITE].rede;
        idxMelhor++;
      }
    } else {
      // Sem melhores válidos ainda, todos viram aleatórios
      for (int i = INICIO_MUTACOES; i < NUM_INDIVIDUOS; i++) {
        h_individuos[i] = RedeNeural();
      }
    }

    // 4. Reset dos contadores de todos os indivíduos para nova rodada
    for (int i = 0; i < NUM_INDIVIDUOS; i++) {
      h_individuos[i].ganho = 0;
      h_individuos[i].perda = 0;
      h_individuos[i].rodadasSemApostar = 0;
      h_individuos[i].bvivo = true;
    }

    // Manda os indivíduos atualizados para a memória da GPU
    cudaMemcpy(d_individuos, h_individuos, sizeIndividuos,
               cudaMemcpyHostToDevice);

    // Inicializa os pesos dos indivíduos aleatórios
    // Se não há melhores válidos, inicializa todos (0 a NUM_INDIVIDUOS)
    // Caso contrário, inicializa apenas de idxIndividuo até INICIO_MUTACOES
    if (numMelhoresValidos == 0) {
      // Primeira geração ou sem sobreviventes: inicializa todos
      iniciarPesos<<<grid, bloco>>>(d_individuos, NUM_INDIVIDUOS, d_estados);
    } else {
      // Inicializa apenas os aleatórios (de idxIndividuo até INICIO_MUTACOES)
      int numAleatorios = INICIO_MUTACOES - idxIndividuo;
      if (numAleatorios > 0) {
        dim3 gridAleatorios(numAleatorios);
        dim3 blocoAleatorios(32);
        iniciarPesosRange<<<gridAleatorios, blocoAleatorios>>>(
            d_individuos, idxIndividuo, INICIO_MUTACOES, d_estados);
      }
    }
    cudaDeviceSynchronize();

    // Chama kernel de mutação nos indivíduos (INICIO_MUTACOES em diante)
    // Só muta se há melhores válidos para copiar
    if (numMelhoresValidos > 0) {
      mutarPesosDevice<<<grid, bloco>>>(d_individuos, INICIO_MUTACOES,
                                        NUM_INDIVIDUOS, d_estados);
      cudaDeviceSynchronize();
    }

    std::cout << "Taxa de vitoria do melhor individuo: " << h_melhores10[9].taxa
              << "%" << std::endl;

    // Verifica se a meta foi atingida
    if (h_melhores10[9].taxa >= META_TAXA_VITORIA) {
      bmetaAtingida = true;
      std::cout << "Meta atingida! Taxa: " << h_melhores10[9].taxa << "%"
                << std::endl;
    }

    // A nova rodada de inferência (treinamento) começa automaticamente
    // no próximo ciclo do while, não precisa de código adicional aqui
  }

  free(h_candles);
  delete[] h_individuos;
  delete[] h_melhores10;
  cudaFree(d_candles);
  cudaFree(d_individuos);
  cudaFree(d_estados);

  return 0;
}