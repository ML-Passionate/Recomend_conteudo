#
#
#  Sistema de recomendação utilizando redes neurais, 
#  baseado em embeddings de usuários e filmes. Uiliza o conjunto de dados MovieLens 
#  para treinar um modelo que prevê classificações que um usuário daria a um filme. 
#  Dados divididos em treino e validação, arquitetura de rede neural com Keras, treina o modelo, 
#  avalia o desem-penho e gera gráficos com os resultados.
#
#  dar pip install pickle5
#

#%%
import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Lambda
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import HTML, display, Markdown
import tabulate
from utils import *
from utils import get_user_vecs


pd.set_option("display.precision", 1)


#%%
#
#  O dataset deriva do MovieLens ml-latest-small - https://grouplens.org/conjunto de dadoss/movielens/latest/
#

top10_df = pd.read_csv("data/content_top10_df.csv")
bygenre_df = pd.read_csv("data/content_bygenre_df.csv")
top10_df

#%%
#
# a próxima tabela mostra a informação em ordem de genre
#

bygenre_df


#%%
#
# Carregar Data, set configuration variables
#
#

item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

num_user_features = user_train.shape[1] - 3  # remove o userid, contagem de rating count e rating médio durante o treinamento 
num_item_features = item_train.shape[1] - 1  # remove o movie id durante o treinamento
uvs = 3  # inicio do vetor de usuario tipo de filme
ivs = 3  # inicio do vetor de tipo de filme
u_s = 3  # inicio das colunas usadas no treinamento, usuário
i_s = 1  # inicio das colunas usadas no treinamento, items
print(f"Numero de vetores de treinamento: {len(item_train)}")

#%%
#  Array de treinamento
#

tabela_texto_user = pprint_train_text(user_train, user_features, uvs,  u_s, maxcount=5)

print(tabela_texto_user)

#%%

tabela_texto_item = pprint_train_text(item_train, item_features, ivs, i_s, maxcount=5, user=False)

print(tabela_texto_item)


#%%

#   Acima, o array de filmes contém o ano em que o filme foi lançado, a avaliação média e um indicador 
#   para cada gênero possível. O indicador é igual a um para cada gênero que se aplica ao filme.
#   O ID do filme não é usado no treinamento, mas é útil na interpretação dos dados.

print(f"y_train[:5]: {y_train[:5]}")


#%%
#
# scale training 
#
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled    = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
# ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))

#%%
#
#   Para permitir que possamos avaliar os resultados, nós iremos dividir os dados em conjuntos de treino e teste, 
#   nós iremos usar o train_test_split do sklearn para dividir e embaralhar os dados. 
#   Note que definir o estado inicial aleatório com o mesmo valor garante que item, user e y sejam embaralhados de forma idêntica.


item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test       = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)

print(f"Tamanho dos dados de treinamento: {item_train.shape}")
print(f"Tamanho dos dados de teste: {item_test.shape}")

#%%

# Os dados escalados e embaralhados agora têm média igual a zero.


pprint_train_text(user_train, user_features, uvs, u_s, maxcount=5)



#%%

#
# Construindo uma rede neural, duas redes que são combinadas por um produto escalar (dot product).
#
#
#  Use um modelo sequencial do Keras
#
#  * A primeira camada é uma camada densa com 256 unidades e ativação *relu*.
#  * A segunda camada é uma camada densa com 128 unidades e ativação *relu*.
#  * A terceira camada é uma camada densa com `num_outputs` unidades e ativação linear ou nenhuma ativação.
#
#

num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(num_outputs),])

item_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(num_outputs),])

# crie o usuário e aponte a para a rede base

input_user = tf.keras.layers.Input(shape=(num_user_features,))

vu = user_NN(input_user)
# vu = tf.linalg.l2_normalize(vu, axis=1)
vu = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vu)


# crie item e aponte para a rede base
input_item = tf.keras.layers.Input(shape=(num_item_features,))
vm = item_NN(input_item)
# vm = tf.linalg.l2_normalize(vm, axis=1)
vm = Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm)


# calcule o produto escalar dos vetores vu e 

output = tf.keras.layers.Dot(axes=1)([vu, vm])

# especifique a entrada e saida do modelo
model = tf.keras.Model([input_user, input_item], output)

model.summary()

#%%
#
# layer denso com o relu    
#    
#

usarr_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'),])

item_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'),])
 
usarr_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(num_outputs),])

item_NN = tf.keras.models.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(128, activation='relu'),tf.keras.layers.Dense(num_outputs),])

#%%

#
#   iremos usar o erro médio quadratico e o otimizador Adam
#

tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)

#%%
#
# roda as  = 30

# 
#
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

#%%
#
#   vamos avaliar o modelo para determinar perda nos dados de teste
#

model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

# é comparável ao erro de treinamento o que confirma que não houve over-fitting

#%%
#
#   inferencias para um novo usuário. Este usuário gosta de aventura e fantasia
#
new_user_id = 5000
new_rating_ave = 0.0
new_action = 0.0
new_adventure = 5.0
new_animation = 0.0
new_childrens = 0.0
new_comedy = 0.0
new_crime = 0.0
new_documentary = 0.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 0.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])

#%%

# gera e replica o vetor do usuário para match o numero de filmes no dataset
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# Escalar os vetores de usuarios e itens
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# faz a inferência
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# ordena os resultados
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

# filmes em HTML
print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)

#%%
#
#  inferencia para um usuário já existente
#

uid = 2 
# vetor de usuário
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# escala os vetores de usuarios e itens
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# faz a inferencia
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# Ordena os resultados
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]

# Imprime os resultados em 

print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)

#%%
"""
A rede neural acima produz dois vetores de características: um vetor de características do usuário
e um vetor de características do filme. Esses vetores possuem 32 entradas cujos valores são difíceis 
de interpretar diretamente. No entanto, itens semelhantes terão vetores semelhantes. 
Essa informação pode ser usada para fazer recomendações. 
Por exemplo, se um usuário avaliou muito bem o filme "Toy Story 3", 
é possível recomendar filmes semelhantes selecionando aqueles com vetores de características de filme parecidos.

"""
#%%
#
#  função que calcula a distância euclidiana entre dois vetores
#

def sq_dist(a,b):
    
    d = sum((a_i - b_i)**2 for a_i, b_i in zip(a, b))
    
    return d

#%%
#
#
#
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])
print(f"Distancia quadratica euclidiana entre a1 e b1: {sq_dist(a1, b1):0.3f}")
print(f"Distancia quadratica euclidiana entre a2 e b2: {sq_dist(a2, b2):0.3f}")
print(f"Distancia quadratica euclidiana entre a3 e b3: {sq_dist(a3, b3):0.3f}")


#%%
#
'''
Uma matriz de distâncias entre filmes pode ser calculada uma vez quando o modelo é treinado e, 
em seguida, reutilizada para novas recomendações sem retreinamento. O primeiro passo, 
após o treinamento do modelo, é obter o vetor de características do filme Vm, para cada um dos filmes. Para isso, usaremos o item_NN treinado e construiremos um pequeno modelo que nos permita passar os vetores dos filmes por ele para gerar Vm

'''

input_item_m = tf.keras.layers.Input(shape=(num_item_features,))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.keras.layers.Lambda(lambda x: tf.linalg.l2_normalize(x, axis=1))(vm_m)
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()

#%%
#
#  Vamos agora calcular uma matriz das distâncias quadradas entre cada vetor de características do filme e todos os outros vetores de características de filmes:
#
#

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"Tamanho de todos os vetores de atributos de todos os filmes: {vms.shape}")


#%%
#
#  Podemos então encontrar o filme mais próximo identificando o valor mínimo ao longo de cada linha. 
#  Utilizaremos arrays mascarados do NumPy para evitar selecionar o mesmo filme. Os valores mascarados na 
#  diagonal não serão incluídos no cálculo.
#

count = 50  # numero de filmes para listar
dim = len(vms)
dist = np.zeros((dim,dim))

for i in range(dim):
    for j in range(dim):
        dist[i,j] = sq_dist(vms[i, :], vms[j, :])
        
m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i,0])
    movie2_id = int(item_vecs[min_idx,0])
    disp.append( [movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                  movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
               )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")
table
