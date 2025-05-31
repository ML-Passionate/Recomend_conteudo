# Content Recommendation with Neural Networks and TensorFlow

![image](https://github.com/user-attachments/assets/ea128b77-af19-4a14-98a4-6b971b003b8c)

Recommendation system using neural networks, based on user and movie embeddings. Uses the MovieLens dataset to train a model that predicts ratings a user would give to a movie. Data is split into training and validation, neural network architecture with Keras, trains the model, evaluates performance, and generates result graphs.

### Descriptive List of Program Functionality

#### **1. Data Loading and Preprocessing**
- **Data Source**: Uses the MovieLens dataset (`ml-latest-small`), loading movie information (such as `content_top10_df.csv` and `content_bygenre_df.csv`).
- **Data Structure**:
  - `item_train`: Movie features (year, genres, etc.).
  - `user_train`: User features (genre preferences, rating history).
  - `y_train`: Actual user ratings.
- **Preprocessing**:
  - Data normalization with `StandardScaler` (for users and movies) and `MinMaxScaler` (for ratings).
  - Splits data into training (80%) and testing (20%) using `train_test_split`.

---

#### **2. Neural Network Model Architecture**
- **Two Parallel Neural Networks**:
  - **User Network (`user_NN`)**:
    - Layers: `Dense(256, relu)` → `Dense(128, relu)` → `Dense(32)`.
    - Output: 32-dimensional vector normalized with `L2-normalization`.
  - **Movie Network (`item_NN`)**:
    - Layers: `Dense(256, relu)` → `Dense(128, relu)` → `Dense(32)`.
    - Output: 32-dimensional vector normalized with `L2-normalization`.
- **Combination**:
  - The outputs of the two networks are combined via **dot product** to predict the user's rating.

---

#### **3. Training and Evaluation**
- **Compilation**:
  - Loss function: `Mean Squared Error` (MSE).
  - Optimizer: `Adam` with learning rate 0.01.
- **Training**:
  - 30 epochs with training data.
- **Evaluation**:
  - Measures loss (MSE) on test data to check for overfitting.

---

#### **4. Recommendation System**
- **Inference for New Users**:
  - Generates a feature vector for a new user with defined preferences (e.g., adventure and fantasy).
  - Uses the trained model to predict ratings for all movies.
  - Recommends the top 10 movies with the highest predicted ratings.
- **Inference for Existing Users**:
  - Retrieves the user's rating history.
  - Generates personalized recommendations based on the model's predictions.
- **Similarity-Based Recommendations**:
  - Computes **squared Euclidean distance** between movie feature vectors.
  - Identifies similar movies by masking the distance matrix diagonal to avoid self-comparison.

---

#### **5. Visualization and Interpretation**
- **Tables and Charts**:
  - Displays formatted tables with `tabulate` to show recommended movies and their features.
  - Functions like `print_pred_movies` and `print_existing_user` format output in HTML for better readability.
- **Embedding Interpretation**:
  - The 32-dimensional vectors (embeddings) capture latent patterns of users and movies, enabling similarity-based recommendations.

---

#### **6. Additional Features**
- **Inverse Scaling**:
  - Reverts prediction normalization to interpret ratings on the original scale.
- **Distance Masking**:
  - Uses `numpy.ma` to ignore the diagonal of the distance matrix, ensuring a movie is not compared to itself.

---

#### **7. Dependencies and Configurations**
- **Libraries**:
  - `TensorFlow/Keras` for neural networks.
  - `Pandas` and `NumPy` for data manipulation.
  - `scikit-learn` for preprocessing and data splitting.
- **Configurations**:
  - Fixed random seeds (`random_state=1`, `tf.random.set_seed(1)`) for reproducibility.

---
