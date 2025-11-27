
// função que normaliza o preço para usar no treinamento da rede
function normalizar5(a, b, c, d) {
    const fator = a;      // sempre usa o primeiro como referência
    return [
        a / fator,
        b / fator,
        c / fator,
        d / fator,
    ];
}

console.log(normalizar5(82580.08000000, 82580.08000000, 82550.00000000, 82550.01000000));