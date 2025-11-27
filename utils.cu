struct BTCNormalizado{
    float abertura;
    float minima;
    float maxima;
    float fechamento;
};

/** função que normaliza os dados do bitcoin
 * @param abertura abertura do candle
 * @param minima minima do candle
 * @param maxima maxima do candle
 * @param fechamento fechamento do candle
 * @return uma struct com os valores normalizados
 * @note chamavel pelo host e pelo device
 */
__host__ __device__ BTCNormalizado normalizarBTC(float abertura,float minima, float maxima, float fechamento){
    float factor = abertura;
    BTCNormalizado btc;

    btc.abertura = abertura / factor;
    btc.minima = minima / factor;
    btc.maxima = maxima / factor;
    btc.fechamento = fechamento / factor;

    return btc;
    
};
