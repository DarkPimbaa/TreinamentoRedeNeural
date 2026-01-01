#pragma once

#include "./types.hpp"
#include <chrono>
// #include "nlohmann/json.hpp"

// using json = nlohmann::json;

enum class TimeUnit { us, ms, s, m, h, d };

class Timer {
public:
    Timer(TimeUnit unit = TimeUnit::ms) 
        : m_unit(unit), m_stopped(false) {
        m_start = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        m_stopped = true;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start).count();

        switch (m_unit) {
            case TimeUnit::us: return (double)duration;
            case TimeUnit::ms: return duration / 1000.0;
            case TimeUnit::s:  return duration / 1000000.0;
            case TimeUnit::m:  return duration / 60000000.0;
            case TimeUnit::h:  return duration / 3600000000.0;
            case TimeUnit::d:  return duration / 86400000000.0;
            default: return 0.0;
        }
    }

    void reset() {
        m_start = std::chrono::high_resolution_clock::now();
        m_stopped = false;
    }

private:
    TimeUnit m_unit;
    bool m_stopped;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

void lerCSV_mallocc(const char *caminho, Candle *dados, size_t tamanho_maximo);

/**
 * @brief cria um arquivo csv com suporte e resistencia dos ultimos 15 minutos, 30 minutos, 1 hora e 3 horas totalizando 8 novos valores por linha
 * Importante! as primeiras linhas do arquivo original vão ser sacrificadas a fim de calcular as medias do novo arquivo
 * 
 * @param csv - arquivo original com os dados brutos do bitcoin
 * @param tamanho - tamanho do arquivo original em linhas sem contar a linha de cabeçalho
 * @param novoArquivo - nome do novo arquivo
 */
void attData(const char* csv, int tamanho, const char* novoArquivo);