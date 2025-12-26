#include "includes/imgui/imgui.h"
#include "includes/imgui-sfml/imgui-SFML.h"

#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Rect.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Shape.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>
#include <cstddef>
#include <cstdio>
#include <SFML/System/Angle.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include "includes/types.hpp"
#include "includes/utils.hpp"
using json = nlohmann::json;
/* #region structs*/
enum Decisao{
    COMPROU,
    VENDEU,
    NADA
};

struct Entrada{
    int indiceDaEntrada;
    Decisao decisao;
};
#ifndef __CUDA_ARCH__
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Entrada, indiceDaEntrada, decisao);
#endif
struct Backlog{
    int idxAtual = 0;
    int quantidadeDeEntradas = 0;
    Entrada entradas[1000];
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Backlog, idxAtual, quantidadeDeEntradas, entradas);
/* #endregion*/

struct SpritCandle{
    sf::RectangleShape body;
    sf::RectangleShape wick;
};

class Plotter{
public:
    SpritCandle candlesSprit[250]; // Buffer para 201 candles (100 tras + 1 atual + 100 frente)
    sf::ConvexShape seta; // Agora ConvexShape para fazer uma seta desenhada
    int itemsToDraw = 0;
    
    void prepararCandles(int idxAtual, Candle *candles, int totalCandles, Decisao decidiu){
        int windowWidth = 1000;
        int windowHeight = 1000;
        float marginV = 50.0f; // Margem vertical
        float drawHeight = windowHeight - 2 * marginV;

        int viewRange = 100; // Quantos candles para tras e para frente
        int startIdx = idxAtual - viewRange;
        int endIdx = idxAtual + viewRange;

        // Limites do array
        if (startIdx < 0) startIdx = 0;
        
        // 1. Calcular Min/Max price na janela
        float minPrice = candles[startIdx].minima;
        float maxPrice = candles[startIdx].maxima;

        for(int i = startIdx; i <= endIdx; i++){
            if(candles[i].minima < minPrice) minPrice = candles[i].minima;
            if(candles[i].maxima > maxPrice) maxPrice = candles[i].maxima;
        }

        if(maxPrice == minPrice) maxPrice += 0.0001f; // Evitar div/0

        float scaleY = drawHeight / (maxPrice - minPrice);
        float candleWidth = (float)windowWidth / (float)(endIdx - startIdx + 1);
        // Deixar um espacinho entre candles
        float spacing = 1.0f;
        float bodyWidth = candleWidth - spacing;
        if(bodyWidth < 1.0f) bodyWidth = 1.0f;

        itemsToDraw = 0;

        for(int i = startIdx; i <= endIdx; i++){
            int spritIdx = i - startIdx;
            if (spritIdx >= 250) break; 

            float openY  = marginV + (maxPrice - candles[i].abertura) * scaleY;
            float closeY = marginV + (maxPrice - candles[i].fechamento) * scaleY;
            float highY  = marginV + (maxPrice - candles[i].maxima) * scaleY;
            float lowY   = marginV + (maxPrice - candles[i].minima) * scaleY;

            float x = spritIdx * candleWidth;

            // Configurar Wick
            candlesSprit[spritIdx].wick.setSize({1, lowY - highY});
            candlesSprit[spritIdx].wick.setPosition({x + bodyWidth / 2, highY});
            
            // Configurar Corpo
            float bodyHeight = std::abs(closeY - openY);
            if(bodyHeight < 1.0f) bodyHeight = 1.0f; // Altura minima
            
            candlesSprit[spritIdx].body.setSize({bodyWidth, bodyHeight});
            candlesSprit[spritIdx].body.setPosition({x, std::min(openY, closeY)});

            // Cor
            sf::Color color = (candles[i].fechamento >= candles[i].abertura) ? sf::Color::Green : sf::Color::Red;
            candlesSprit[spritIdx].body.setFillColor(color);
            candlesSprit[spritIdx].wick.setFillColor(color);

            // Seta de decisao
            if(i == idxAtual){
                if (decidiu != Decisao::NADA) {
                    seta.setPointCount(7);
                    // Dimensões da seta
                    float arrowH = 20.0f;
                    float arrowW = 14.0f;
                    float shaftW = 6.0f;
                    float headH = 10.0f; 
                    float shaftH = arrowH - headH;
                    float wingW = (arrowW - shaftW) / 2.0f;

                    if (decidiu == Decisao::COMPROU) {
                        // Seta apontando para CIMA
                        // Pontos em relacao ao topo-esquerdo da bounding box da seta
                        // 0: fim shaft esq, 1: inicio shaft esq (onde encosta na cabeca), 2: ponta asa esq, 3: topo, 4: ponta asa dir, 5: inicio shaft dir, 6: fim shaft dir
                        
                        seta.setPoint(0, {wingW, arrowH});           // Bottom-Left of Shaft
                        seta.setPoint(1, {wingW, headH});            // Shaft meets Head Left
                        seta.setPoint(2, {0, headH});                // Left Wing Tip
                        seta.setPoint(3, {arrowW / 2, 0});           // Top Tip
                        seta.setPoint(4, {arrowW, headH});           // Right Wing Tip
                        seta.setPoint(5, {wingW + shaftW, headH});   // Shaft meets Head Right
                        seta.setPoint(6, {wingW + shaftW, arrowH});  // Bottom-Right of Shaft

                        // Posicionar abaixo do candle, apontando pra cima
                        seta.setOrigin({arrowW / 2, 0}); // Origem na ponta (topo)
                        seta.setPosition({x + bodyWidth/2, lowY + 5}); // 5px de padding abaixo do Low
                        seta.setFillColor(sf::Color::Yellow);

                    } else { // VENDEU
                        // Seta apontando para BAIXO
                        seta.setPoint(0, {wingW, 0});                // Top-Left of Shaft
                        seta.setPoint(1, {wingW, shaftH});           // Shaft meets Head Left
                        seta.setPoint(2, {0, shaftH});               // Left Wing Tip
                        seta.setPoint(3, {arrowW / 2, arrowH});      // Bottom Tip
                        seta.setPoint(4, {arrowW, shaftH});          // Right Wing Tip
                        seta.setPoint(5, {wingW + shaftW, shaftH});  // Shaft meets Head Right
                        seta.setPoint(6, {wingW + shaftW, 0});       // Top-Right of Shaft

                        // Posicionar acima do candle, apontando pra baixo
                        seta.setOrigin({arrowW / 2, arrowH}); // Origem na ponta (fundo)
                        seta.setPosition({x + bodyWidth/2, highY - 5}); // 5px de padding acima do High
                        seta.setFillColor(sf::Color::Magenta);
                    }
                } else {
                    seta.setPointCount(0);
                }
            }
            itemsToDraw++;
        }
    };

    void desenharCandles(sf::RenderWindow &window){
        for (int i = 0; i < itemsToDraw; i++) {
            window.draw(candlesSprit[i].wick);
            window.draw(candlesSprit[i].body);
        }
        if(seta.getPointCount() > 0)
            window.draw(seta);
    };

};

int main() {
    sf::RenderWindow window(sf::VideoMode({1000, 1000}), "NN-Plotter");
    window.setFramerateLimit(60);
    bool b = ImGui::SFML::Init(window);

    /* #region backlogLoad*/
    Backlog *backlog = new Backlog();
    std::ifstream f("backlog.json");
    json j;
    f >> j;
    *backlog = j;
    int totalBacklogs = backlog->quantidadeDeEntradas;
    int idxentradaAtual = 0;
    /* #endregion */


    /* #region CandlesLoad */
    Candle *candles = new Candle[964750];
    lerCSV_malloccNovo("./btc_windicadores.csv", candles, 964750);
    /* #endregion */

   Plotter *plotter = new Plotter();

    //* MAIN LOOP
    sf::Clock deltaClock;
    while (window.isOpen()) {
        while (const auto event = window.pollEvent()) {
            ImGui::SFML::ProcessEvent(window, *event);

            if (event->is<sf::Event::Closed>()) {
                window.close();
            }
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Geral");

        if (ImGui::Button("<-")) {
            if (idxentradaAtual == 0) {
            
            }else {
                idxentradaAtual--;
            }
        };

        ImGui::SameLine();

        if (ImGui::Button("->")){
            if (idxentradaAtual == totalBacklogs) {
            
            }else {
                idxentradaAtual++;
            }
        };

        ImGui::Text("Entrada %d de %d", idxentradaAtual, totalBacklogs);

        ImGui::End();

        plotter->prepararCandles(backlog->entradas[idxentradaAtual].indiceDaEntrada, candles, 964750, backlog->entradas[idxentradaAtual].decisao);

        window.clear();
        plotter->desenharCandles(window);
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
}