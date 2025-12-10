#include "./includes/types.hpp"
#include "./includes/utils.hpp"
#include "./includes/corretora.cpp"
#include <cstddef>
#include <ostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <iostream>

// macros
#define NUM_ENTRADAS 60
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2
#define NUM_INDIVIDUOS 100

// --- GLOBAIS ---

// instancia as redes
std::vector<RedeNeural> redes(NUM_INDIVIDUOS, RedeNeural());

// vetor com os melhores individuos DA GERAÇÃO ATUAL
std::vector<RedeNeural> melhores10(10, RedeNeural());

// [NOVO] vetor com as 10 melhores redes DE TODO O TREINAMENTO
std::vector<RedeNeural> melhores10TodosOsTempos;

// instancia a corretora
Corretora corretora = Corretora();


// Comparador para a Priority Queue (Max Heap)
// Retorna true se A deve ficar "abaixo" de B na pilha
struct ComparadorPQ {
    bool operator()(const RedeNeural& a, const RedeNeural& b) const {
        if (a.ganho != b.ganho)
            return a.ganho < b.ganho;  // Menor ganho fica embaixo (maior vai pro topo)
        return a.perda > b.perda;      // Se ganho igual, maior perda fica embaixo (menor perda vai pro topo)
    }
};

// [NOVO] Comparador para ordenação padrão (std::sort)
// Retorna true se A é MELHOR que B (para ordenar do melhor para o pior)
bool eMelhor(const RedeNeural& a, const RedeNeural& b) {
    if (a.ganho != b.ganho) return a.ganho > b.ganho; // Maior ganho primeiro
    return a.perda < b.perda;                         // Menor perda primeiro
}

// retorna as 10 melhores redes neurais da geração atual
std::vector<RedeNeural> obterMelhores10(std::vector<RedeNeural>& redes) {
    std::priority_queue<RedeNeural, std::vector<RedeNeural>, ComparadorPQ> pq;
    
    for (const auto& rede : redes) {
        pq.push(rede);
    }
    
    std::vector<RedeNeural> top10;
    top10.reserve(10);
    
    int count = 0;
    while (!pq.empty() && count < 10) {
        top10.push_back(pq.top());
        pq.pop();
        ++count;
    }
    
    return top10;  // já vem ordenado do melhor para o 10º melhor
}

// [NOVO] Função para gerenciar o Ranking Global (Hall da Fama)
void atualizarMelhoresTodosOsTempos(const RedeNeural& campeaDaGeracao) {
    // Adiciona a campeã na lista
    melhores10TodosOsTempos.push_back(campeaDaGeracao);

    // Ordena do melhor para o pior
    std::sort(melhores10TodosOsTempos.begin(), melhores10TodosOsTempos.end(), eMelhor);

    // Se tiver mais de 10, remove os piores (o que sobrou no final da lista)
    if (melhores10TodosOsTempos.size() > 10) {
        melhores10TodosOsTempos.resize(10);
    }
}

// [MODIFICADO] Repovoa misturando Melhores da Geração + Melhores de Todos os Tempos
void repovoarComEliteMista(std::vector<RedeNeural>& melhoresGeracao, std::vector<RedeNeural>& melhoresGlobal, std::vector<RedeNeural>& populacao)
{
    // Cria um vetor de pais contendo a união dos dois grupos
    std::vector<RedeNeural> pais;
    pais.reserve(melhoresGeracao.size() + melhoresGlobal.size());

    // Adiciona os melhores desta geração
    pais.insert(pais.end(), melhoresGeracao.begin(), melhoresGeracao.end());

    // Adiciona os melhores de todos os tempos
    pais.insert(pais.end(), melhoresGlobal.begin(), melhoresGlobal.end());

    // Segurança: se não houver pais (primeira rodada bugada?), não faz nada
    if (pais.empty()) return;

    const size_t popSize = populacao.size();
    const size_t qtdPais = pais.size();
    
    // Distribui os genes dos pais para a população inteira (Round Robin)
    for (size_t i = 0; i < popSize; ++i) {
        // Usa o operador módulo (%) para rodar ciclicamente entre os pais disponíveis (sejam 10, 15 ou 20)
        size_t idxPai = i % qtdPais;

        populacao[i].setRede(pais[idxPai].getRede());
        populacao[i].ganho = 0;
        populacao[i].perda = 0;
    }
}

int main(){
    for (size_t geracao = 0; geracao < 10000; geracao++) { // Aumentei para loop infinito ou grande

        // iniciar o loop de apostas
        for (size_t i = 0; i < 10000; i++) {
            std::vector<float> entrada;
            
            if (corretora.historicoNormalizado.size() < 10) {
                std::cout << "Erro: histórico insuficiente." << std::endl;
                return 1;
            }

            // Cria o vetor de entrada
            for (size_t k = 0; k < 10; k++) {
                // Ajuste de índice: pega os últimos candles disponíveis se o loop 'i' não estiver sincronizado com o histórico
                // Assumindo que você quer sempre os ultimos 10 baseados num offset ou fixo. 
                // Mantive sua lógica original de 0 a 9, mas cuidado: isso pega sempre os mesmos 10 candles estáticos se o histórico não andar.
                const Candle& c = corretora.historicoNormalizado[k]; 
                entrada.push_back(c.abertura);
                entrada.push_back(c.maxima);
                entrada.push_back(c.minima);
                entrada.push_back(c.fechamento);
                entrada.push_back(c.volume);
                entrada.push_back(c.trades);
            }

            // Processamento das redes
            for (size_t r = 0; r < redes.size(); r++) {
                Resultado resultado = redes[r].iniciar(entrada);

                bool s1 = resultado.resultado[0];
                bool s2 = resultado.resultado[1];

                if (s1 && !s2) {
                    corretora.apostar(r, true); // Compra
                } else if (!s1 && s2) {
                    corretora.apostar(r, false); // Venda
                }
                // Casos s1==s2 (true/true ou false/false) não fazem nada
            }

            corretora.gerarResultado(redes);
        }

        // --- FIM DA GERAÇÃO ---

        // 1. Pega os 10 melhores da geração atual
        melhores10 = obterMelhores10(redes);

        // 2. Atualiza o ranking global (Hall da Fama) apenas com o CAMPEÃO desta geração
        if (!melhores10.empty()) {
            atualizarMelhoresTodosOsTempos(melhores10[0]);
        }

        // 3. Repovoa a população usando os (até) 20 indivíduos de elite
        repovoarComEliteMista(melhores10, melhores10TodosOsTempos, redes);

        // 4. Aplica mutação na nova geração (exceto talvez nos clones puros se quiser elitismo puro, mas aqui aplica em todos)
        for (RedeNeural& rede : redes) {
            rede.mutacao();
        }

        system("clear");

        float media = 0.0f;
        if ((melhores10[0].ganho + melhores10[0].perda) > 0) {
            media = ((float)melhores10[0].ganho / (melhores10[0].ganho + melhores10[0].perda)) * 100;
        }

        std::cout << "=== GERAÇÃO " << geracao << " ===" << std::endl;
        std::cout << "Melhor da Geração - Ganhos: " << melhores10[0].ganho 
                  << " | Perdas: " << melhores10[0].perda 
                  << " | Taxa de Acerto: " << media << "%" << std::endl;

        if (!melhores10TodosOsTempos.empty()) {
            std::cout << "Melhor de Todos os Tempos - Ganhos: " << melhores10TodosOsTempos[0].ganho 
                      << " | Perdas: " << melhores10TodosOsTempos[0].perda << std::endl;
        }
    }
    return 0;
}