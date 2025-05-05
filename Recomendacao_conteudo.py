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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import HTML, display, Markdown
import tabulate
from utils import *

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
"""
Above, the movie array contains the year the film was released, the average rating and an indicator for each potential genre. The indicator is one for each genre that applies to the movie. The movie id is not usard in training but is usarful when interpreting the data.
"""

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
"""
The scaled, shuffled data now has a mean of zero.
"""

#%%
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)

#%%
"""
<a name="4"></a>
#### 4 - Neural Network for content-based filtering
Now, let's construct a neural network as described in the figure above. It iremos have two networks that are combined by a dot product. You iremos construct the two networks. In this example, they iremos be identical. Note that these networks do not need to be the same. If the usarr content was substantially larger than the movie content, você might elect to increase the complexity of the usarr network relative to the movie network. In this case, the content is similar, so the networks are the same.

<a name="ex01"></a>
###### Exercise 1

- Use a Keras sequential modelo
    - The first layer is a dense layer with 256 units and a relu activation.
    - The second layer is a dense layer with 128 units and a relu activation.
    - The third layer is a dense layer with `num_outputs` units and a linear or no activation.   
    
The remainder of the network iremos be provided. The provided code does not usar the Keras sequential modelo but instead usars the Keras [functional api](https://keras.io/guides/functional_api/). Este format allows for more flexibility in how components are interconnected.

"""

#%%
# GRADED_CELL
# UNQ_C1

num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
# ## START CODE HERE ###
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),  
# ## END CODE HERE ###
])

item_NN = tf.keras.models.Sequential([
# ## START CODE HERE ###
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_outputs),
# ## END CODE HERE ###
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()

#%%
# Public tests
from public_tests import *
test_tower(user_NN)
test_tower(item_NN)

#%%
"""
<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
  You can create a dense layer with a relu activation as shown.
    
```python     
usarr_NN = tf.keras.modelos.Sequential([
    ###### START CODE HERE ######     
  tf.keras.layers.Dense(256, activation='relu'),

    
    ###### END CODE HERE ######  
])

item_NN = tf.keras.modelos.Sequential([
    ###### START CODE HERE ######     
  tf.keras.layers.Dense(256, activation='relu'),

    
    ###### END CODE HERE ######  
])
```    
<details>
    <summary><font size="2" color="darkblue"><b> Click for solution</b></font></summary>
    
```python 
usarr_NN = tf.keras.modelos.Sequential([
    ###### START CODE HERE ######     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs),
    ###### END CODE HERE ######  
])

item_NN = tf.keras.modelos.Sequential([
    ###### START CODE HERE ######     
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_outputs),
    ###### END CODE HERE ######  
])
```
</details>
</details>

    

"""

#%%
"""
We iremos usar a mean squared error loss and an Adam optimizer.
"""

#%%
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)

#%%
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

#%%
"""
Evaluate the modelo to determine loss on the test data. 
"""

#%%
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

#%%
"""
It is comparable to the training loss indicating the modelo has not substantially overfit the training data.
"""

#%%
"""
<a name="5"></a>
#### 5 - Predictions
Below, você'll usar vocêr modelo to make predictions in a number of circumstances. 
<a name="5.1"></a>
###### 5.1 - Predictions for a new usarr
First, nós'll create a new usarr and have the modelo suggest movies for that usarr. After você have tried this on the example usarr content, feel free to change the usarr content to match vocêr own preferences and see what the modelo suggests. Note that ratings are betnósen 0.5 and 5.0, inclusive, in half-step increments.
"""

#%%
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
"""
The new usarr enjoys movies from the adventure, fantasy genres. Let's find the top-rated movies for the new usarr.  
Below, nós'll usar a set of movie/item vectors, `item_vecs` that have a vector for each movie in the training/test set. Este is matched with the new usarr vector above and the scaled vectors are usard to predict ratings for all the movies.
"""

#%%
# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 10)

#%%
"""
<a name="5.2"></a>
###### 5.2 - Predictions for an existing usarr.
Let's look at the predictions for "usarr 2", one of the usarrs in the data set. We can compare the predicted ratings with the modelo's ratings.
"""

#%%
uid = 2 
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
sorted_ypu   = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display
sorted_user  = user_vecs[sorted_index]
sorted_y     = y_vecs[sorted_index]

# print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount = 50)

#%%
"""
The modelo prediction is generally within 1 of the actual rating though it is not a very accurate predictor of how a usarr rates specific movies. Este is especially true if the usarr rating is significantly different than the usarr's genre average. You can vary the usarr id above to try different usarrs. Not all usarr id's nósre usard in the training set.
"""

#%%
"""
<a name="5.3"></a>
###### 5.3 - Finding Similar Items
The neural network above produces two feature vectors, a usarr feature vector $v_u$, and a movie feature vector, $v_m$. These are 32 entry vectors whose values are difficult to interpret. Honósver, similar items iremos have similar vectors. Este information can be usard to make recommendations. For example, if a usarr has rated "Toy Story 3" highly, one could recommend similar movies by selecting movies with similar movie feature vectors.

A similarity measure is the squared distance betnósen the two vectors $ \mathbf{v_m^{(k)}}$ and $\mathbf{v_m^{(i)}}$ :
$$\left\Vert \mathbf{v_m^{(k)}} - \mathbf{v_m^{(i)}}  \right\Vert^2 = \sum_{l=1}^{n}(v_{m_l}^{(k)} - v_{m_l}^{(i)})^2\tag{1}$$
"""

#%%
"""
<a name="ex02"></a>
###### Exercise 2

Write a function to compute the square distance.
"""

#%%
# GRADED_FUNCTION: sq_dist
# UNQ_C2
def sq_dist(a,b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """
# ## START CODE HERE ###
    d = sum((a_i - b_i)**2 for a_i, b_i in zip(a, b))
# ## END CODE HERE ###
    return d

#%%
a1 = np.array([1.0, 2.0, 3.0]); b1 = np.array([1.0, 2.0, 3.0])
a2 = np.array([1.1, 2.1, 3.1]); b2 = np.array([1.0, 2.0, 3.0])
a3 = np.array([0, 1, 0]);       b3 = np.array([1, 0, 0])
print(f"squared distance between a1 and b1: {sq_dist(a1, b1):0.3f}")
print(f"squared distance between a2 and b2: {sq_dist(a2, b2):0.3f}")
print(f"squared distance between a3 and b3: {sq_dist(a3, b3):0.3f}")

#%%
"""
**Expected Output**:

squared distance betnósen a1 and b1: 0.000    
squared distance betnósen a2 and b2: 0.030   
squared distance betnósen a3 and b3: 2.000
"""

#%%
# Public tests
test_sq_dist(sq_dist)

#%%
"""
<details>
  <summary><font size="3" color="darkgreen"><b>Click for hints</b></font></summary>
    
  While a summation is often an indication a for loop should be usard, here the subtraction can be element-wise in one statement. Further, você can utilized np.square to square, element-wise, the result of the subtraction. np.sum can be usard to sum the squared elements.
    
</details>

    

"""

#%%
"""
A matrix of distances betnósen movies can be computed once when the modelo is trained and then reusard for new recommendations without retraining. The first step, once a modelo is trained, is to obtain the movie feature vector, $v_m$, for each of the movies. To do this, nós iremos usar the trained `item_NN` and build a small modelo to allow us to run the movie vectors through it to generate $v_m$.
"""

#%%
input_item_m = tf.keras.layers.Input(shape=(num_item_features))    # input layer
vm_m = item_NN(input_item_m)                                       # use the trained item_NN
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)                        # incorporate normalization as was done in the original model
model_m = tf.keras.Model(input_item_m, vm_m)                                
model_m.summary()

#%%
"""
Once você have a movie modelo, você can create a set of movie feature vectors by using the modelo to predict using a set of item/movie vectors as input. `item_vecs` is a set of all of the movie vectors. It must be scaled to usar with the trained modelo. The result of the prediction is a 32 entry feature vector for each movie.
"""

#%%
scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")

#%%
"""
Let's now compute a matrix of the squared distance betnósen each movie feature vector and all other movie feature vectors:
<figure>
    <left> <img src="./images/distmatrix.PNG"   style="width:400px;height:225px;" ></center>
</figure>
"""

#%%
"""
We can then find the closest movie by finding the minimum along each row. We iremos make usar of [numpy masked arrays](https://numpy.org/doc/1.21/usarr/tutorial-ma.html) to avoid selecting the same movie. The masked values along the diagonal won't be included in the computation.
"""

#%%
count = 50  # number of movies to display
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

#%%
"""
The results show the modelo iremos generally suggest a movie with similar genre's.
"""

#%%
"""
<a name="6"></a>
#### 6 - Congratulations! <img align="left" src="./images/film_award.png" style=" width:40px;">
You have completed a content-based recommender system.    

Este structure is the basis of many commercial recommender systems. The usarr content can be greatly expanded to incorporate more information about the usarr if it is available.  Items are not limited to movies. Este can be usard to recommend any item, books, cars or items that are similar to an item in vocêr 'shopping cart'.
"""

#%%
"""
<details>
  <summary><font size="2" color="darkgreen"><b>Please click here if você want to experiment with any of the non-graded code.</b></font></summary>
    <p><i><b>Important Note: Please only do this when você've already passed the assignment to avoid problems with the autograder.</b></i>
    <ol>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
        <li> Hit the “Edit Metadata” button next to the code cell which você want to lock/unlock</li>
        <li> Set the attribute value for “editable” to:
            <ul>
                <li> “true” if você want to unlock it </li>
                <li> “false” if você want to lock it </li>
            </ul>
        </li>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
    </ol>
    <p> Here's a short demo of how to do the steps above: 
        <br>
        <img src="https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center" alt="unlock_cells.gif">
</details>
"""