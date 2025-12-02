#pragma once
#include <string>
#include <vector>
#include "types.hpp"


// le o arquivo csv e retorna uma array com todos os candles
std::vector<Candle> lerCSV(const std::string& caminho);
