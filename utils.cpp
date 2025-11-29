#include "utils.hpp"
#include <fstream>
#include <sstream>

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
