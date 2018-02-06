Ruidos são causados por erros na transmissão de dados. Os pixels corrompidos ou são alterados para valores muito baixos ou para valores muito baixos, causando diferenças bruscas de tons entre os pixels e seus vizinhos. Alguns ruídos apresentem essa característica muito intensa e espalhada pela imagem, são os denominados ruídos ‘salt & pepper’, outros já apresentam aparência com ‘chiados’, estes são ruídos gaussianos.

Para a remoção destes ruídos, são utilizados filtros espaciais. No presente trabalho, são utilizados 3 tipos de filtros, e comparados seus rendimentos e resultados em cada tipo de ruído, gaussiano e salt and pepper.



Podemos notar que os filtros da media e gaussianos são mais eficazes em imagens com ruído gaussiano, já o filtro da mediana tem sua eficácia máxima em imagens com o ruído salt and pepper. Isto ocorre pois o filtro da media e gaussiano ao encontrar ruídos salt and pepper acabam gerando um valor para o pixel, muito diferente de seu valor original, devido a sua implementação, já o filtro da mediana não gera tais modificações tão drásticas.

