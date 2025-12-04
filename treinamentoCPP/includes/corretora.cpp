#pragma once

#include "./types.hpp"
#include "utils.hpp"
#include <cstddef>
#include <vector>

#ifndef NUM_ENTRADAS
// Se NUM_ENTRADAS AINDA NÃO estiver definido...
#define NUM_ENTRADAS 60
// ...agora o definimos com 60
#endif

struct Aposta{
    int incideDoApostador;
    bool compraOuVenda;
};

class Corretora {
public:    
    std::vector<Candle> candle;
    std::vector<Candle> historico;
    std::vector<Candle> historicoNormalizado;
    std::vector<Aposta> apostas;
    
    // inicia o histórico de candles
    Corretora(){
        candle = lerCSV("./filtrado.csv");
        for (size_t i = 0; i < NUM_ENTRADAS / 10; i++) {
            historico[i] = candle[i];
        }

        historicoNormalizado = normalizarCandle(historico);
    };

    // função que cada rede vai chamar se for apostar
    void apostar(int indiceDoApostador, bool compraOuVenda){
        Aposta aposta;
        aposta.compraOuVenda = compraOuVenda;
        aposta.incideDoApostador = indiceDoApostador;
        apostas.push_back(aposta);
    }

    //TODO criar função que vai atualizar o histórico

    //TODO criar função que retornar o resultado da aposta para os apostados(ver quem ganhou e quem perdeu) e atualizar o histórico.

    //TODO função que vai zerar a lista de apostas


};
