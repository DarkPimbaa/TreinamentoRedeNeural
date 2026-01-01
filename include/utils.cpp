#include "./utils.hpp"
#include <fstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>

void attData(const char* csv, int tamanho, const char* novoArquivo) {
    std::ifstream input(csv);
    std::ofstream output(novoArquivo);
    
    if (!input.is_open() || !output.is_open()) return;

    std::string line, header;
    std::getline(input, header);

    std::vector<Candle> dados;
    dados.reserve(tamanho);

    // Parse rápido
    while (std::getline(input, line)) {
        std::stringstream ss(line);
        std::string val;
        Candle c;
        
        auto getNext = [&ss]() { 
            std::string v; 
            if(!std::getline(ss, v, ',')) return 0.0f;
            return std::stof(v); 
        };
        
        c.abertura = getNext();
        c.maxima = getNext();
        c.minima = getNext();
        c.fechamento = getNext();
        c.volume = getNext();
        c.trades = getNext();
        
        dados.push_back(c);
    }

    output << header << ",s15,r15,s30,r30,s60,r60,s180,r180\n";

    const int janelas[] = {15, 30, 60, 180};
    const int max_janela = 180;

    for (int i = max_janela; i < dados.size(); ++i) {
        // Dados originais
        output << dados[i].abertura << "," << dados[i].maxima << "," << dados[i].minima << ","
               << dados[i].fechamento << "," << dados[i].volume << "," << (int)dados[i].trades;

        // Suporte e Resistência (min/max do período)
        for (int j : janelas) {
            float s = std::numeric_limits<float>::max();
            float r = std::numeric_limits<float>::lowest();

            for (int k = i - j; k < i; ++k) {
                if (dados[k].minima < s) s = dados[k].minima;
                if (dados[k].maxima > r) r = dados[k].maxima;
            }
            output << "," << s << "," << r;
        }
        output << "\n";
    }

    input.close();
    output.close();
}

void lerCSV_mallocc(const char *caminho, Candle *dados, size_t tamanho_maximo) {
  std::ifstream file(caminho);
  if (!file.is_open()) return;

  std::string linha;
  std::getline(file, linha); // Ignora cabeçalho

  size_t i = 0;
  while (std::getline(file, linha) && i < tamanho_maximo) {
      std::stringstream ss(linha);
      std::string coluna;

      auto parseNext = [&ss, &coluna]() {
          if (std::getline(ss, coluna, ',')) return std::stof(coluna);
          return 0.0f;
      };

      // Colunas originais
      dados[i].abertura   = parseNext();
      dados[i].maxima     = parseNext();
      dados[i].minima     = parseNext();
      dados[i].fechamento = parseNext();
      dados[i].volume     = parseNext();
      dados[i].trades     = parseNext();

      // Colunas de Suporte e Resistência (os 8 novos valores)
      dados[i].s15  = parseNext();
      dados[i].r15  = parseNext();
      dados[i].s30  = parseNext();
      dados[i].r30  = parseNext();
      dados[i].s60  = parseNext();
      dados[i].r60  = parseNext();
      dados[i].s180 = parseNext();
      dados[i].r180 = parseNext();

      i++;
  }
  file.close();
};