#pragma once

#include <cstddef>
#include <vector>
// NUNCA faça #include de .cpp em header → remova "./utils.cpp"
// #include "./types.hpp" só se realmente precisar de tipos definidos lá

#define NUM_ENTRADAS 60
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2


struct Neuronio {
    float valor = 0.f;
    float peso[NUM_ENTRADAS] = { 0.f};        // recomendo float, não int
};

struct Camada {
    Neuronio neuronio[NUM_ENTRADAS];
};

struct Rede {
    float entrada[NUM_ENTRADAS];
    Camada oculto[NUM_CAMADAS];
    Neuronio saida[NUM_SAIDAS];      // normalmente só 1 camada de saída
};

struct Resultado{
    bool resultado[NUM_SAIDAS] = {false};
};

class RedeNeural {
public:
    Rede rede;
    Resultado resultado;

    // Construtor que inicializa TUDO de uma vez
    RedeNeural();

    // função que iterar sobre todos os neuronios e vai trazer um resultado
    Resultado iniciar(std::vector<float> valores){
        if (valores.size() != NUM_ENTRADAS) {
        // Tratar erro, lançar exceção ou retornar resultado vazio
        return Resultado(); 
        }

        //TODO criar função que zera os valores de entrada, camadas ocultas e saida.

        // loop que inicia os valores dos neuronios de entrada
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
            rede.entrada[neuronio] = valores[neuronio];
        }


        // loop quer vai pegar os valores da camada de entrada e vai alimentar a primeira camada oculta
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
            for (size_t neuronioPrev = 0; neuronioPrev < NUM_ENTRADAS; neuronioPrev++) {
                rede.oculto[0].neuronio[neuronio].valor += rede.entrada[neuronioPrev] * rede.oculto[0].neuronio[neuronio].peso[neuronioPrev];
            }

            //função de ativação
            if (rede.oculto[0].neuronio[neuronio].valor < 0.f) {
                rede.oculto[0].neuronio[neuronio].valor = 0.f;
            }
        }

        // loop que vai iterar nas camadas ocultas a partir da segunda
        for (size_t camada = 1; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                for (size_t neuronioPrev = 0; neuronioPrev < NUM_ENTRADAS; neuronioPrev++) {
                    rede.oculto[camada].neuronio[neuronio].valor += rede.oculto[camada - 1].neuronio[neuronioPrev].valor * rede.oculto[camada].neuronio[neuronio].peso[neuronioPrev];
                }

                //função de ativação
                if (rede.oculto[camada].neuronio[neuronio].valor < 0.f) {
                    rede.oculto[camada].neuronio[neuronio].valor = 0.f;
                }
            }
        }

        // loop que vai iterar sobre os neuronios de saida
        for (size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++) {
                for (size_t neuronioPrev = 0; neuronioPrev < NUM_ENTRADAS; neuronioPrev++) {
                    rede.saida[neuronio].valor += rede.oculto[NUM_CAMADAS - 1].neuronio[neuronioPrev].valor * rede.saida[neuronio].peso[neuronioPrev];
                }

                //função de ativação
                if (rede.saida[neuronio].valor < 0.f) {
                    rede.saida[neuronio].valor = 0.f;
                }
            }


        for (size_t resultado = 0; resultado < NUM_SAIDAS; resultado++) {

            if (rede.saida[resultado].valor > 0.f) {
            this->resultado.resultado[resultado] = true;
            }else{
            this->resultado.resultado[resultado] = false;
            }
            
        }

        return this->resultado;
        

    }

    //TODO criar função de inicializar os pesos aleatoriamente

    //TODO criar funções de set e get
    
};