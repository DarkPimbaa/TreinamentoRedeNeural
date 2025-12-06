#pragma once

#include "./types.hpp"
#include "utils.hpp"
#include <cstddef>
#include <vector>
#include "./redeNeural.cpp"

#ifndef NUM_ENTRADAS
// Se NUM_ENTRADAS AINDA NÃO estiver definido...
#define NUM_ENTRADAS 60 //
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
    int indice = -1;
    
    // inicia o histórico de candles
    Corretora(){
        candle = lerCSV("./filtrado.csv");
        for (size_t i = 0; i < NUM_ENTRADAS / 6; i++) {
            historico.push_back(candle[i]);
            indice++;
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

    // atualiza o historico normal e o historico normalizado
    void attHistorico(){

        for (size_t i = 0; i < historico.size() - 1; i++) {
            historico[i] = historico[i + 1];
        }
        
        historico[historico.size() - 1] = candle[indice];
        historicoNormalizado = normalizarCandle(historico);
    }

    // gera o resultado e atualiza o histórico
    void gerarResultado(std::vector<RedeNeural>& redes){

        bool resultado;
        if (indice + 1 >= candle.size()) {
            // Finalizar simulação ou retornar
            return; 
        }
        if (candle[indice].abertura > candle[indice + 1].abertura ) {
            resultado = false;
        }else {
            resultado = true;
        }
        indice++;
        attHistorico();

        // for que itera sobre todas as apostas
        for (size_t i = 0; i < apostas.size(); i++) {
            
            if (apostas[i].compraOuVenda == true && resultado == true) {
                redes[apostas[i].incideDoApostador].ganho++;
            
            }else if (apostas[i].compraOuVenda == false && resultado == false) {
                redes[apostas[i].incideDoApostador].ganho++;
            }else{
                redes[apostas[i].incideDoApostador].perda++;
            }
        }
        apostas.clear();
    }


};
