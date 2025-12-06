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

// instancia as redes
std::vector<RedeNeural> redes(NUM_INDIVIDUOS, RedeNeural());

// vetor com os melhores individuos
std::vector<RedeNeural> melhores10(10, RedeNeural());

// instancia a corretora
Corretora corretora = Corretora();


// usado na função obterMelhores10
struct Comparador {
    bool operator()(const RedeNeural& a, const RedeNeural& b) const {
        if (a.ganho != b.ganho)
        return a.ganho < b.ganho;  // maior ganho vem primeiro
    return a.perda > b.perda;    // se ganho igual, menor perda vem primeiro
}
};

// retorna as 10 melhores redes neurais
std::vector<RedeNeural> obterMelhores10(std::vector<RedeNeural>& redes) {
    std::priority_queue<RedeNeural, std::vector<RedeNeural>, Comparador> pq;
    
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

// repovoa o vetor de redes com as 10 melhores redes
void repovoarComMelhores(std::vector<RedeNeural>& melhores10, std::vector<RedeNeural>& redes)
{
    const size_t pop = redes.size();
    const size_t nMelhores = melhores10.size();  // deve ser 10
    
    for (size_t i = 0; i < pop; ++i) {
        size_t idxOrigem = i / 10;                   // 0..9  → 0
        // 10..19 → 1
                                                     // 20..29 → 2
                                                     // etc.
        if (idxOrigem >= nMelhores) idxOrigem = nMelhores - 1; // segurança
        
        redes[i].setRede(melhores10[idxOrigem].getRede());
        redes[i].ganho = 0;
        redes[i].perda = 0;   // ou .perdas, dependendo do nome real
    }
}

//TODO criar o loop principal de treinamento
int main(){
for (size_t geracao = 0; geracao < 10; geracao++) {


    // iniciar o loop de apostas
for (size_t i = 0; i < 100; i++) {
    std::vector<float> valores;
    
    // carrega o vetor valores com os valores em float dos candles.
    for (size_t i = 0; i < corretora.historicoNormalizado.size(); i++) {
        int ii = i * 6;
        valores.push_back(corretora.historicoNormalizado[ii].abertura);
        valores.push_back(corretora.historicoNormalizado[ii + 1].maxima);
        valores.push_back(corretora.historicoNormalizado[ii + 2].minima);
        valores.push_back(corretora.historicoNormalizado[ii + 3].fechamento);
        valores.push_back(corretora.historicoNormalizado[ii + 4].volume);
        valores.push_back(corretora.historicoNormalizado[ii + 5].trades);
    }



    // cada rede verifica se vai apostar ou não
    for (size_t rede = 0; rede < redes.size(); rede++) {
        Resultado resultado = redes[rede].iniciar(valores);

        if (resultado.resultado[0] == true && resultado.resultado[1] == true) {
            // não faz nada
        
        }else if(resultado.resultado[0] == false && resultado.resultado[1] == false){
            // não faz nada
        }else if (resultado.resultado[0] == true && resultado.resultado[1] == false) {
            //compra
            corretora.apostar(rede, true);
        }else if (resultado.resultado[0] == false && resultado.resultado[1] == true) {
            // vende
            corretora.apostar(rede, false);
        }
    }

    // corretora da o resultado
    corretora.gerarResultado(redes);

}

// pega as 10 melhores redes
melhores10 = obterMelhores10(redes);
repovoarComMelhores(melhores10, redes);

// aplica mutação a nova geração
for (RedeNeural rede : redes) {
    rede.mutacao();
}
system("cls");

std::cout << "melhor individuo da geracao: " << geracao << std::endl;
std::cout << "ganhos: " << melhores10[0].ganho << std::endl;
std::cout << "ganhos: " << melhores10[0].ganho << std::endl;
std::cout << "media: : " << ((melhores10[0].ganho / (melhores10[0].ganho + melhores10[0].perda)) * 100.0) << std::endl;
}
    return 0;
}