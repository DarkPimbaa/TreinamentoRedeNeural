#pragma once

#include <cstddef>
#include <vector>
#include "./utils.hpp"

#define NUM_ENTRADAS 60
#define NUM_CAMADAS 5
#define NUM_SAIDAS 2


struct Neuronio {
    float valor = 0.f;
    float peso[NUM_ENTRADAS] = { 0.f};
};

struct Camada {
    Neuronio neuronio[NUM_ENTRADAS];
    //Neuronio bias; //TODO implementar o calculo de bias nas camadas ocultas
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
    int ganho = 0;
    int perda = 0;

    // Construtor que inicializa TUDO de uma vez
    RedeNeural();

    // função que itera sobre todos os neuronios e retorna um resultado
    Resultado iniciar(std::vector<float> valores){
        if (valores.size() != NUM_ENTRADAS) {
        // Tratar erro, lançar exceção ou retornar resultado vazio
        return Resultado(); 
        }

        //zera os valores de toda a rede neural e o resultado
        zeraValores();

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


    /** zera os valores de toda a rede neural e resultado*/
    void zeraValores(){
        
        // zera os neuronios de entrada
        for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
            rede.entrada[neuronio] = 0.f;
        }

        // zera os valores das camadas ocultas
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                rede.oculto[camada].neuronio[neuronio].valor = 0.f;
            }
        }

        // zera os valores dos neuronios de saida
        for (size_t neuronio = 0 ; neuronio < NUM_SAIDAS; neuronio++) {
            rede.saida[neuronio].valor = 0.f;
        }

        // zera o resultado
        for (size_t i = 0; i < NUM_SAIDAS; i++) {
            resultado.resultado[i] = false;
        }

    };
    
    // inicializa os pesos dos neuronios das camadas ocultas e camada de saida
    void iniciarPesos(){

        //! talvez seja preciso fazer um cast (float) porque Rand::float retorna um double

        // inicia os pesos de camada de saida
        for(size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++){
            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                rede.saida[neuronio].peso[peso] = Rand::Float(-1.f, 1.f);
            }
        };

        // inicia os pesos das camadas ocultas
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            for (size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++) {
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                    rede.oculto[camada].neuronio[neuronio].peso[peso] = Rand::Float(-1.f, 1.f);
                }
            }
        }

    };

    // tem 10% de chance de mutar cada peso aleatóriamente para cima ou para baixo por 0.1
    void mutacao(){
        
        // muta os pesos dos neuronios de saida limitando entre -1 e 1
        for(size_t neuronio = 0; neuronio < NUM_SAIDAS; neuronio++){
            for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                if (Rand::Int(0, 10) == 1) {
                    if (Rand::Int( 0, 1) == 1) {
                        rede.saida[neuronio].peso[peso] += 0.1f;
                        if (rede.saida[neuronio].peso[peso] > 1.f) {
                            rede.saida[neuronio].peso[peso] = 1.f;
                        }
                    }else{
                        rede.saida[neuronio].peso[peso] += -0.1f;
                        if (rede.saida[neuronio].peso[peso] < -1.f) {
                            rede.saida[neuronio].peso[peso] = -1.f;
                        }
                    }
                }
            }
        };

        // muta os pesos das camadas ocultas
        for (size_t camada = 0; camada < NUM_CAMADAS; camada++) {
            for(size_t neuronio = 0; neuronio < NUM_ENTRADAS; neuronio++){
                for (size_t peso = 0; peso < NUM_ENTRADAS; peso++) {
                    if (Rand::Int(0, 10) == 1) {
                        if (Rand::Int( 0, 1) == 1) {
                            rede.oculto[camada].neuronio[neuronio].peso[peso] += 0.1;
                            if (rede.oculto[camada].neuronio[neuronio].peso[peso] > 1.f) {
                                rede.oculto[camada].neuronio[neuronio].peso[peso] = 1.f;
                            }
                        }else{
                            rede.oculto[camada].neuronio[neuronio].peso[peso] += -0.1;
                            if (rede.oculto[camada].neuronio[neuronio].peso[peso] < -1.f) {
                                rede.oculto[camada].neuronio[neuronio].peso[peso] = -1.f;
                            }
                        }
                    }
                }
            };
        }

    }

    void setRede(Rede rede){
        this->rede = rede;
    };

    Rede getRede(){
        return this->rede;
    };
    
};