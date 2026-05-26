# entrega dio

este projeto monta uma recomendação visual simples usando tensorflow e keras.

em vez de treinar uma rede do zero, foi usada a mobilenetv2 pré-treinada para extrair embeddings das imagens. esses embeddings são vetores que resumem características visuais, como forma, cor e textura.

depois disso, o sistema compara os vetores usando cosine similarity e retorna as imagens mais parecidas. o foco foi entender o fluxo de embeddings + similarity search de um jeito simples e funcional.
