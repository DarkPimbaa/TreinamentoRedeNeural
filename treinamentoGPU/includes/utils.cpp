#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "./types.hpp"
#include "./utils.hpp"

std::vector<Candle> lerCSV(const std::string &caminho) {
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

/**
 * @brief Carrega dados de um arquivo CSV diretamente na memória alocada por
 * malloc.
 * * @param caminho O caminho para o arquivo CSV.
 * @param dados O ponteiro para o bloco de memória alocado (Candle*).
 * @param tamanho_maximo O número máximo de Candles que o bloco 'dados' pode
 * armazenar.
 */
void lerCSV_mallocc(const char *caminho, Candle *dados, size_t tamanho_maximo) {
  std::ifstream file(caminho);

  if (!file.is_open()) {
    return;
  }

  std::string linha;
  std::getline(file, linha); // Ignora o cabeçalho

  size_t i = 0;
  while (std::getline(file, linha) && i < tamanho_maximo) {
    std::stringstream ss(linha);
    std::string coluna;

    // Usamos a notação de array (dados[i]) que é sintaticamente idêntica
    // à aritmética de ponteiros (p.ex., *(dados + i)) quando se trabalha com
    // malloc.

    std::getline(ss, coluna, ',');
    dados[i].abertura = std::stof(coluna);

    std::getline(ss, coluna, ',');
    dados[i].maxima = std::stof(coluna);

    std::getline(ss, coluna, ',');
    dados[i].minima = std::stof(coluna);

    std::getline(ss, coluna, ',');
    dados[i].fechamento = std::stof(coluna);

    std::getline(ss, coluna, ',');
    dados[i].volume = std::stof(coluna);

    std::getline(ss, coluna, ',');
    dados[i].trades = std::stof(coluna);

    i++;
  }
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

// Escolhe um elemento aleatório de qualquer container (vector, array,
// string...)
template <typename Container>
auto Rand::Choice(const Container &c) -> decltype(auto) {
  std::uniform_int_distribution<size_t> dist(0, c.size() - 1);
  return c[dist(gen)];
};

// normaliza um vetor de candles baseado na abertura do ultimo candle
std::vector<Candle> normalizarCandle(std::vector<Candle> candles) {
  float factor = candles.back().abertura;

  for (auto &c : candles) {
    c.abertura /= factor;
    c.maxima /= factor;
    c.minima /= factor;
    c.fechamento /= factor;
  };

  return candles;
};

