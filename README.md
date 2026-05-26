# image-recommendation-dio

um projeto simples de recomendação por similaridade visual usando deep learning.

a ideia é escolher uma imagem de produto e encontrar outras imagens parecidas no dataset. para isso, o projeto usa uma rede pré-treinada para transformar cada imagem em um vetor de características, e depois compara esses vetores.

o foco aqui não foi treinar uma ia do zero. foi montar um fluxo pequeno, funcional e fácil de entender usando embeddings e busca por similaridade.

## como funciona

o projeto segue este caminho:

1. carrega algumas imagens de produtos
2. usa a mobilenetv2 pre-treinada para extrair caracteristicas visuais
3. transforma cada imagem em um embedding
4. compara a imagem de entrada com as imagens do dataset
5. retorna as mais parecidas usando cosine similarity
6. mostra o resultado em um grid com matplotlib

as categorias do dataset são bem simples:

- relógios
- tenis
- camisetas

## embeddings, sem complicar

um embedding é um vetor de números que representa uma imagem.

em vez de comparar imagem por imagem olhando pixel por pixel, a rede neural cria uma espécie de resumo visual. esse resumo guarda sinais como formato, cor, textura e partes importantes da imagem.

duas imagens visualmente parecidas tendem a gerar vetores parecidos.

## cosine similarity

cosine similarity é uma forma de medir o quanto dois vetores apontam para uma direção parecida.

no projeto, ela ajuda a responder uma pergunta simples:

> essa imagem parece com aquela?

quanto maior o valor da similaridade, mais próximas as imagens estão no espaço dos embeddings.

## estrutura do projeto

```text
image-recommendation-dio/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── image_recommendation_colab.ipynb
├── src/
│   └── recommendation.py
├── dataset/
│   ├── watches/
│   ├── shoes/
│   ├── shirts/
│   └── README.md
├── images/
│   ├── input/
│   ├── output/
│   └── README.md
├── README-assets/
└── docs/
    └── entrega_dio.md
```

## como rodar pelo github e google colab

depois de subir o projeto para o github:

1. abra o arquivo `notebooks/image_recommendation_colab.ipynb` no github
2. clique em `open in colab`, se o botão aparecer
3. se preferir, use este formato de link:

```text
https://colab.research.google.com/github/fezleep/image-recommendation-dio/blob/main/notebooks/image_recommendation_colab.ipynb
```

o notebook já está preparado para clonar o repositório dentro do colab quando o dataset não estiver disponível no ambiente.

## como rodar localmente

crie um ambiente virtual, instale as dependências e rode o notebook.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/image_recommendation_colab.ipynb
```

se estiver no linux ou mac:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/image_recommendation_colab.ipynb
```

## prints do projeto

espaço reservado para colocar prints reais depois de rodar o notebook.

exemplo:

```md
![resultado](README-assets/result.png)
```

## aprendizados

com esse projeto eu pratiquei:

- uso de uma cnn pré-treinada com tensorflow e keras
- extração de embeddings de imagens
- comparação de vetores com cosine similarity
- montagem de um fluxo simples de recomendação visual
- organização de um projeto para github e google colab

## proximos passos

algumas melhorias que dá para fazer depois:

- usar imagens reais de produtos
- aumentar o dataset
- salvar os embeddings para nao recalcular tudo sempre
- testar outras redes pre-treinadas
- criar uma interface simples com streamlit

## conclusao

este projeto não treina uma rede do zero.

foi utilizada uma rede pré-treinada para extração das características visuais. o foco aqui foi montar um fluxo simples de recomendação visual, usando embeddings e similarity search de um jeito direto e fácil de acompanhar.

é um projeto pequeno, mas já mostra bem a ideia principal por trás de muitos sistemas de recomendação por imagem.
