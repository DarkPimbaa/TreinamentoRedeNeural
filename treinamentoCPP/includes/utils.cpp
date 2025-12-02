#pragma once

#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <random>


std::vector<Candle> lerCSV(const std::string& caminho) {
    std::ifstream file(caminho);
    std::vector<Candle> dados;
    std::string linha;

    std::getline(file, linha);

    while (std::getline(file, linha)) {
        std::stringstream ss(linha);
        std::string coluna;
        Candle c;

        std::getline(ss, coluna, ',');
        c.abertura = std::stof(coluna);

        std::getline(ss, coluna, ',');
        c.maxima = std::stof(coluna);

        std::getline(ss, coluna, ',');
        c.minima = std::stof(coluna);

        std::getline(ss, coluna, ',');
        c.fechamento = std::stof(coluna);

        std::getline(ss, coluna, ',');
        c.volume = std::stof(coluna);

        std::getline(ss, coluna, ',');
        c.trades = std::stof(coluna);

        dados.push_back(c);
    }

    return dados;
}


class Rand {
    inline static std::random_device rd;
    inline static std::mt19937 gen{rd()};
    
public:
    // Inteiro entre min e max (inclusive)
    static int Int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
    // Double entre min e max (max exclusivo por padrão, como em Python)
    static double Float(double min = 0.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
    
    // Escolhe um elemento aleatório de qualquer container (vector, array, string...)
    template<typename Container>
    static auto Choice(const Container& c) -> decltype(auto) {
        std::uniform_int_distribution<size_t> dist(0, c.size() - 1);
        return c[dist(gen)];
    }
};
