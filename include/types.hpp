#pragma once

typedef struct Candle{
    float abertura;
    float maxima;
    float minima;
    float fechamento;
    float volume;
    float trades;
    float s15, r15, s30, r30, s60, r60, s180, r180;
} Candle;