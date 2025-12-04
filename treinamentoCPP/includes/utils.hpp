#pragma once

#include <string>
#include <vector>
#include "types.hpp"
#include <random>


// le o arquivo csv e retorna uma array com todos os candles
std::vector<Candle> lerCSV(const std::string& caminho);

class Rand {
    inline static std::random_device rd;
    inline static std::mt19937 gen{rd()};
public:
    // Inteiro entre min e max (inclusive)
    static int Int(int min, int max);
    // Double entre min e max (max exclusivo por padrão, como em Python)
    static double Float(double min = 0.0, double max = 1.0);
    
    // Escolhe um elemento aleatório de qualquer container (vector, array, string...)
    template<typename Container>
    static auto Choice(const Container& c) -> decltype(auto);
};

//retorna o vetor de candles normalizado baseado na abetura do candle no ultimo indice do vetor
std::vector<Candle> normalizarCandle(std::vector<Candle> candles);