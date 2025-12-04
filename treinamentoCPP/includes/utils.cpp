#include "utils.hpp"
#include <fstream>
#include <sstream>
#include <vector>


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


    // Inteiro entre min e max (inclusive)
    int Rand::Int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    };
    
    // Double entre min e max (max exclusivo por padrão, como em Python)
    double Rand::Float(double min, double max) {
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    };
    
    // Escolhe um elemento aleatório de qualquer container (vector, array, string...)
    template<typename Container>
    auto Rand::Choice(const Container& c) -> decltype(auto) {
        std::uniform_int_distribution<size_t> dist(0, c.size() - 1);
        return c[dist(gen)];
    };

    // normaliza um vetor de candles baseado na abertura do ultimo candle
    std::vector<Candle> normalizarCandle(std::vector<Candle> candles){
        float factor = candles.back().abertura;

        for (auto& c : candles) {
            c.abertura /= factor;
            c.maxima /= factor;
            c.minima /= factor;
            c.fechamento /= factor;
        };

        return candles;
    };

