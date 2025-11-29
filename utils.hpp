#pragma once
#include <string>
#include <vector>
#include "types.hpp"   // ajuste o nome conforme você renomear o types.cu

std::vector<Candle> lerCSV(const std::string& caminho);
