# Recomendação de conteúdo com redes neurais e tensorflow

![image](https://github.com/user-attachments/assets/ea128b77-af19-4a14-98a4-6b971b003b8c)


Sistema de recomendação utilizando redes neurais, baseado em embeddings de usuários e filmes. Uiliza o conjunto de dados MovieLens para treinar um modelo que prevê classificações que um usuário daria a um filme. Dados divididos em treino e validação, arquitetura de rede neural com Keras, treina o modelo, avalia o desempenho e gera gráficos com os resultados.

### Lista Descritiva do Funcionamento do Programa

#### **1. Carregamento e Pré-processamento de Dados**
- **Fonte de Dados**: Utiliza o conjunto de dados MovieLens (`ml-latest-small`), carregando informações de filmes (como `content_top10_df.csv` e `content_bygenre_df.csv`).
- **Estrutura dos Dados**:
  - `item_train`: Características dos filmes (ano, gêneros, etc.).
  - `user_train`: Características dos usuários (preferências por gêneros, histórico de avaliações).
  - `y_train`: Avaliações reais dos usuários.
- **Pré-processamento**:
  - Normalização dos dados com `StandardScaler` (para usuários e filmes) e `MinMaxScaler` (para as avaliações).
  - Divisão dos dados em treino (80%) e teste (20%) usando `train_test_split`.

---

#### **2. Arquitetura do Modelo de Rede Neural**
- **Duas Redes Neurais Paralelas**:
  - **Rede do Usuário (`user_NN`)**:
    - Camadas: `Dense(256, relu)` → `Dense(128, relu)` → `Dense(32)`.
    - Saída: Vetor de 32 dimensões normalizado (`L2-normalization`).
  - **Rede do Filme (`item_NN`)**:
    - Camadas: `Dense(256, relu)` → `Dense(128, relu)` → `Dense(32)`.
    - Saída: Vetor de 32 dimensões normalizado (`L2-normalization`).
- **Combinação**:
  - As saídas das duas redes são combinadas via **produto escalar** (`Dot product`) para prever a avaliação do usuário.

---

#### **3. Treinamento e Avaliação**
- **Compilação**:
  - Função de perda: `Mean Squared Error` (MSE).
  - Otimizador: `Adam` com taxa de aprendizado 0.01.
- **Treinamento**:
  - 30 épocas com os dados de treino.
- **Avaliação**:
  - Medição da perda (MSE) nos dados de teste para verificar overfitting.

---

#### **4. Sistema de Recomendação**
- **Inferência para Novos Usuários**:
  - Gera um vetor de características para um novo usuário com preferências definidas (ex.: aventura e fantasia).
  - Utiliza o modelo treinado para prever avaliações de todos os filmes.
  - Recomenda os 10 filmes com maior predição de avaliação.
- **Inferência para Usuários Existentes**:
  - Recupera o histórico de avaliações do usuário.
  - Gera recomendações personalizadas com base nas predições do modelo.
- **Recomendações Baseadas em Similaridade**:
  - Calcula a **distância euclidiana quadrática** entre os vetores de características dos filmes.
  - Identifica filmes semelhantes mascarando a diagonal da matriz de distâncias para evitar auto-comparação.

---

#### **5. Visualização e Interpretação**
- **Tabelas e Gráficos**:
  - Exibe tabelas formatadas com `tabulate` para mostrar filmes recomendados e suas características.
  - Funções como `print_pred_movies` e `print_existing_user` formatam a saída em HTML para melhor legibilidade.
- **Interpretação dos Embeddings**:
  - Os vetores de 32 dimensões (embeddings) capturam padrões latentes de usuários e filmes, permitindo recomendações baseadas em similaridade.

---

#### **6. Funcionalidades Adicionais**
- **Escalonamento Inverso**:
  - Reverte a normalização das predições para interpretar as avaliações na escala original.
- **Máscara de Distâncias**:
  - Usa `numpy.ma` para ignorar a diagonal da matriz de distâncias, garantindo que um filme não seja comparado consigo mesmo.

---

#### **7. Dependências e Configurações**
- **Bibliotecas**:
  - `TensorFlow/Keras` para redes neurais.
  - `Pandas` e `NumPy` para manipulação de dados.
  - `scikit-learn` para pré-processamento e divisão de dados.
- **Configurações**:
  - Sementes aleatórias fixas (`random_state=1`, `tf.random.set_seed(1)`) para reprodutibilidade.

---
