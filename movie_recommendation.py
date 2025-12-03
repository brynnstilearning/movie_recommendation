"""
SISTEM REKOMENDASI FILM: KNN vs DECISION TREE REGRESSOR
Analisis Kinerja Algoritma K-Nearest Neighbors dan Decision Tree Regressor
untuk Sistem Rekomendasi Film Sederhana

Author: Nur Muhammad Anang Febriananto (230605110103)
Dataset: MovieLens 100K
"""

# ============================================================================
# 1. SETUP & IMPORT LIBRARIES
# ============================================================================
                    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from datetime import datetime
import os  # ← PENTING: Untuk handle path

# PENTING: Pindah ke folder yang benar
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"✓ Working directory: {os.getcwd()}\n")

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)
from sklearn.metrics.pairwise import cosine_similarity

# Visualization settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("SISTEM REKOMENDASI FILM: KNN vs DECISION TREE REGRESSOR")
print("="*70)
print(f"Waktu Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n[STEP 1] LOADING DATASET...")
print("-"*70)

# CATATAN: Download dataset dari https://grouplens.org/datasets/movielens/100k/
# Extract file dan letakkan di folder yang sama dengan notebook ini

# Load ratings data
column_names_ratings = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=column_names_ratings, encoding='latin-1')

# Load movies data
column_names_movies = ['movie_id', 'title', 'release_date', 'video_release_date',
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                       'Thriller', 'War', 'Western']
movies = pd.read_csv('u.item', sep='|', names=column_names_movies, 
                     encoding='latin-1', on_bad_lines='skip')

# Load users data
column_names_users = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=column_names_users, encoding='latin-1')

print(f"✓ Ratings data loaded: {ratings.shape[0]} rows, {ratings.shape[1]} columns")
print(f"✓ Movies data loaded: {movies.shape[0]} rows, {movies.shape[1]} columns")
print(f"✓ Users data loaded: {users.shape[0]} rows, {users.shape[1]} columns")

# Display sample data
print("\n[Sample Data - Ratings]")
print(ratings.head())
print("\n[Sample Data - Movies]")
print(movies[['movie_id', 'title', 'release_date']].head())
print("\n[Sample Data - Users]")
print(users.head())

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "="*70)
print("[STEP 2] EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# 3.1 Basic Statistics
print("\n[3.1] BASIC STATISTICS")
print("-"*70)

print("\n[Ratings Statistics]")
print(ratings.describe())

print(f"\nJumlah pengguna unik: {ratings['user_id'].nunique()}")
print(f"Jumlah film unik: {ratings['movie_id'].nunique()}")
print(f"Jumlah rating total: {len(ratings)}")
print(f"Rating minimum: {ratings['rating'].min()}")
print(f"Rating maksimum: {ratings['rating'].max()}")
print(f"Rata-rata rating: {ratings['rating'].mean():.2f}")
print(f"Median rating: {ratings['rating'].median():.2f}")

# 3.2 Missing Values
print("\n[3.2] MISSING VALUES CHECK")
print("-"*70)
print("Missing values in ratings:")
print(ratings.isnull().sum())
print("\nMissing values in movies:")
print(movies[['movie_id', 'title', 'release_date']].isnull().sum())
print("\nMissing values in users:")
print(users.isnull().sum())

# 3.3 Sparsity Analysis
print("\n[3.3] SPARSITY ANALYSIS")
print("-"*70)
n_users = ratings['user_id'].nunique()
n_movies = ratings['movie_id'].nunique()
n_ratings = len(ratings)
possible_ratings = n_users * n_movies
sparsity = (1 - (n_ratings / possible_ratings)) * 100

print(f"User-Item Matrix Size: {n_users} users x {n_movies} movies")
print(f"Possible ratings: {possible_ratings:,}")
print(f"Actual ratings: {n_ratings:,}")
print(f"Sparsity: {sparsity:.2f}%")

# 3.4 Visualizations
print("\n[3.4] CREATING VISUALIZATIONS...")
print("-"*70)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis - MovieLens 100K', fontsize=16, fontweight='bold')

# Plot 1: Rating Distribution
ax1 = axes[0, 0]
rating_counts = ratings['rating'].value_counts().sort_index()
ax1.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
ax1.set_xlabel('Rating', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Rating Distribution')
ax1.grid(axis='y', alpha=0.3)
for i, v in enumerate(rating_counts.values):
    ax1.text(rating_counts.index[i], v + 1000, str(v), ha='center', fontweight='bold')

# Plot 2: Ratings per User
ax2 = axes[0, 1]
ratings_per_user = ratings.groupby('user_id').size()
ax2.hist(ratings_per_user, bins=50, color='lightcoral', edgecolor='black')
ax2.set_xlabel('Number of Ratings', fontweight='bold')
ax2.set_ylabel('Number of Users', fontweight='bold')
ax2.set_title('Ratings per User Distribution')
ax2.axvline(ratings_per_user.mean(), color='red', linestyle='--', 
            label=f'Mean: {ratings_per_user.mean():.1f}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Ratings per Movie
ax3 = axes[1, 0]
ratings_per_movie = ratings.groupby('movie_id').size()
ax3.hist(ratings_per_movie, bins=50, color='lightgreen', edgecolor='black')
ax3.set_xlabel('Number of Ratings', fontweight='bold')
ax3.set_ylabel('Number of Movies', fontweight='bold')
ax3.set_title('Ratings per Movie Distribution')
ax3.axvline(ratings_per_movie.mean(), color='green', linestyle='--',
            label=f'Mean: {ratings_per_movie.mean():.1f}')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Plot 4: User Demographics (Age)
ax4 = axes[1, 1]
age_counts = users['age'].value_counts().sort_index()
ax4.bar(age_counts.index, age_counts.values, color='plum', edgecolor='black', width=2)
ax4.set_xlabel('Age', fontweight='bold')
ax4.set_ylabel('Number of Users', fontweight='bold')
ax4.set_title('User Age Distribution')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: eda_analysis.png")
plt.show()

# Additional visualization: Top 10 Most Rated Movies
print("\n[Top 10 Most Rated Movies]")
top_movies = ratings.groupby('movie_id').size().sort_values(ascending=False).head(10)
top_movies_info = movies[movies['movie_id'].isin(top_movies.index)][['movie_id', 'title']]
top_movies_info = top_movies_info.merge(top_movies.to_frame('count'), 
                                         left_on='movie_id', right_index=True)
print(top_movies_info.to_string(index=False))

# ============================================================================
# 4. DATA PREPARATION
# ============================================================================

print("\n" + "="*70)
print("[STEP 3] DATA PREPARATION")
print("="*70)

# 4.1 Merge all data
print("\n[4.1] MERGING DATA...")
print("-"*70)

# Merge ratings with users and movies
data = ratings.merge(users, on='user_id', how='left')
data = data.merge(movies[['movie_id', 'title', 'release_date', 'Action', 'Adventure', 
                          'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']], 
                 on='movie_id', how='left')

print(f"✓ Merged data shape: {data.shape}")
print(f"✓ Columns: {list(data.columns)}")

# 4.2 Feature Engineering
print("\n[4.2] FEATURE ENGINEERING...")
print("-"*70)

# Extract release year
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year
data['release_year'].fillna(data['release_year'].median(), inplace=True)

# Create aggregate features
print("Creating aggregate features...")
user_avg_rating = data.groupby('user_id')['rating'].mean().to_dict()
movie_avg_rating = data.groupby('movie_id')['rating'].mean().to_dict()

data['user_avg_rating'] = data['user_id'].map(user_avg_rating)
data['movie_avg_rating'] = data['movie_id'].map(movie_avg_rating)

# Encode categorical variables
print("Encoding categorical variables...")
le_gender = LabelEncoder()
le_occupation = LabelEncoder()

data['gender_encoded'] = le_gender.fit_transform(data['gender'])
data['occupation_encoded'] = le_occupation.fit_transform(data['occupation'])

# Count genres per movie
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
data['num_genres'] = data[genre_columns].sum(axis=1)

print("✓ Feature engineering completed")

# 4.3 Prepare features for Decision Tree Regressor
print("\n[4.3] PREPARING FEATURES FOR DECISION TREE REGRESSOR...")
print("-"*70)

# Select features
feature_columns = ['user_id', 'movie_id', 'age', 'gender_encoded', 'occupation_encoded',
                  'release_year', 'user_avg_rating', 'movie_avg_rating', 'num_genres'] + genre_columns

X = data[feature_columns].copy()
y = data['rating'].copy()

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"\nFeatures used: {feature_columns}")

# Handle any remaining missing values
X.fillna(X.mean(), inplace=True)

# 4.4 Data Splitting
print("\n[4.4] SPLITTING DATA...")
print("-"*70)

# Stratify by binned ratings to maintain distribution
y_binned = pd.cut(y, bins=[0, 2, 3, 4, 5], labels=[0, 1, 2, 3])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_binned
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")
print(f"✓ Split ratio: {len(X_train)/len(X)*100:.1f}% train, {len(X_test)/len(X)*100:.1f}% test")

# Check rating distribution
print("\n[Rating Distribution in Train vs Test]")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index() * 100
test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index() * 100
dist_df = pd.DataFrame({'Train (%)': train_dist, 'Test (%)': test_dist})
print(dist_df)

# 4.5 Feature Scaling (for Decision Tree Regressor)
print("\n[4.5] FEATURE SCALING...")
print("-"*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ============================================================================
# 5. MODEL TRAINING - K-NEAREST NEIGHBORS (KNN)
# ============================================================================

print("\n" + "="*70)
print("[STEP 4] TRAINING K-NEAREST NEIGHBORS (KNN)")
print("="*70)

# 5.1 Create user-item matrix for KNN
print("\n[5.1] CREATING USER-ITEM MATRIX FOR KNN...")
print("-"*70)

# Create pivot table
user_item_matrix = data.pivot_table(
    index='user_id',
    columns='movie_id',
    values='rating',
    fill_value=0
)

print(f"✓ User-Item Matrix shape: {user_item_matrix.shape}")
print(f"  Users: {user_item_matrix.shape[0]}")
print(f"  Movies: {user_item_matrix.shape[1]}")

# 5.2 Grid Search for KNN (SIMPLIFIED)
print("\n[5.2] HYPERPARAMETER TUNING FOR KNN (Grid Search)...")
print("-"*70)

param_grid_knn = {
    'n_neighbors': [10, 20, 30],      # Dikurangi dari 4 jadi 3
    'weights': ['distance'],           # Dikurangi dari 2 jadi 1
    'metric': ['euclidean']            # Dikurangi dari 2 jadi 1
}

print("Parameter grid (SIMPLIFIED for low-memory systems):")
for key, value in param_grid_knn.items():
    print(f"  {key}: {value}")

knn = KNeighborsRegressor(algorithm='brute')

print("\nStarting Grid Search (5-fold CV)...")
print("⚠️ Using n_jobs=2 to avoid memory issues...")
start_time = time.time()

grid_knn = GridSearchCV(
    knn,
    param_grid_knn,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=2,  # ← PENTING: Dikurangi untuk RAM terbatas
    verbose=1
)

grid_knn.fit(X_train_scaled, y_train)

knn_training_time = time.time() - start_time

print(f"\n✓ Grid Search completed in {knn_training_time:.2f} seconds")
print(f"\n[BEST PARAMETERS FOR KNN]")
print(f"  Best params: {grid_knn.best_params_}")
print(f"  Best CV score (neg MSE): {grid_knn.best_score_:.4f}")
print(f"  Best CV RMSE: {np.sqrt(-grid_knn.best_score_):.4f}")

# 5.3 Train final KNN model with best parameters
print("\n[5.3] TRAINING FINAL KNN MODEL...")
print("-"*70)

best_knn = grid_knn.best_estimator_
print("✓ KNN model trained with best parameters")

# ============================================================================
# 6. MODEL TRAINING - DECISION TREE REGRESSOR
# ============================================================================

print("\n" + "="*70)
print("[STEP 5] TRAINING DECISION TREE REGRESSOR")
print("="*70)

# 6.1 Grid Search for Decision Tree (SIMPLIFIED)
print("\n[6.1] HYPERPARAMETER TUNING FOR DECISION TREE (Grid Search)...")
print("-"*70)

param_grid_dt = {
    'max_depth': [10, 15, 20],         # Dikurangi dari 5 jadi 3
    'min_samples_split': [2, 10, 20],  # Dikurangi dari 4 jadi 3
    'min_samples_leaf': [1, 5],        # Dikurangi dari 4 jadi 2
    'criterion': ['squared_error']     # Dikurangi dari 2 jadi 1
}

print("Parameter grid (SIMPLIFIED for low-memory systems):")
for key, value in param_grid_dt.items():
    print(f"  {key}: {value}")

dt = DecisionTreeRegressor(random_state=42)

print("\nStarting Grid Search (5-fold CV)...")
print("⚠️ Using n_jobs=2 to avoid memory issues...")
start_time = time.time()

grid_dt = GridSearchCV(
    dt,
    param_grid_dt,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=2,  # ← PENTING: Dikurangi untuk RAM terbatas
    verbose=1
)

grid_dt.fit(X_train_scaled, y_train)

dt_training_time = time.time() - start_time

print(f"\n✓ Grid Search completed in {dt_training_time:.2f} seconds")
print(f"\n[BEST PARAMETERS FOR DECISION TREE REGRESSOR]")
print(f"  Best params: {grid_dt.best_params_}")
print(f"  Best CV score (neg MSE): {grid_dt.best_score_:.4f}")
print(f"  Best CV RMSE: {np.sqrt(-grid_dt.best_score_):.4f}")

# 6.2 Train final DT model with best parameters
print("\n[6.2] TRAINING FINAL DECISION TREE MODEL...")
print("-"*70)

best_dt = grid_dt.best_estimator_
print("✓ Decision Tree model trained with best parameters")

# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("[STEP 6] MODEL EVALUATION")
print("="*70)

# 7.1 Make predictions
print("\n[7.1] MAKING PREDICTIONS...")
print("-"*70)

# Predictions on test set
knn_predictions = grid_knn.predict(X_test_scaled)
dt_predictions = grid_dt.predict(X_test_scaled)

# Clip predictions to valid rating range [1, 5]
knn_predictions = np.clip(knn_predictions, 1, 5)
dt_predictions = np.clip(dt_predictions, 1, 5)

print("✓ Predictions completed")

# 7.2 Calculate Regression Metrics
print("\n[7.2] REGRESSION METRICS (RMSE, MAE)")
print("-"*70)

# KNN Metrics
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_predictions))
knn_mae = mean_absolute_error(y_test, knn_predictions)

# Decision Tree Metrics
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_predictions))
dt_mae = mean_absolute_error(y_test, dt_predictions)

print("\n[K-NEAREST NEIGHBORS (KNN)]")
print(f"  RMSE: {knn_rmse:.4f}")
print(f"  MAE:  {knn_mae:.4f}")

print("\n[DECISION TREE REGRESSOR]")
print(f"  RMSE: {dt_rmse:.4f}")
print(f"  MAE:  {dt_mae:.4f}")

# 7.3 Calculate Classification Metrics (Precision, Recall, F1)
print("\n[7.3] CLASSIFICATION METRICS (Threshold = 4.0)")
print("-"*70)

# Convert to binary classification (relevant if rating >= 4)
threshold = 4.0

y_test_binary = (y_test >= threshold).astype(int)
knn_pred_binary = (knn_predictions >= threshold).astype(int)
dt_pred_binary = (dt_predictions >= threshold).astype(int)

# KNN Classification Metrics
knn_precision = precision_score(y_test_binary, knn_pred_binary, zero_division=0)
knn_recall = recall_score(y_test_binary, knn_pred_binary, zero_division=0)
knn_f1 = f1_score(y_test_binary, knn_pred_binary, zero_division=0)

# Decision Tree Classification Metrics
dt_precision = precision_score(y_test_binary, dt_pred_binary, zero_division=0)
dt_recall = recall_score(y_test_binary, dt_pred_binary, zero_division=0)
dt_f1 = f1_score(y_test_binary, dt_pred_binary, zero_division=0)

print("\n[K-NEAREST NEIGHBORS (KNN)]")
print(f"  Precision: {knn_precision:.4f}")
print(f"  Recall:    {knn_recall:.4f}")
print(f"  F1-Score:  {knn_f1:.4f}")

print("\n[DECISION TREE REGRESSOR]")
print(f"  Precision: {dt_precision:.4f}")
print(f"  Recall:    {dt_recall:.4f}")
print(f"  F1-Score:  {dt_f1:.4f}")

# 7.4 Create Comparison Table
print("\n[7.4] METRICS COMPARISON TABLE")
print("="*70)

results_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'RMSE', 'MAE', 'Training Time (s)'],
    'KNN': [
        f"{knn_precision:.4f}",
        f"{knn_recall:.4f}",
        f"{knn_f1:.4f}",
        f"{knn_rmse:.4f}",
        f"{knn_mae:.4f}",
        f"{knn_training_time:.2f}"
    ],
    'Decision Tree': [
        f"{dt_precision:.4f}",
        f"{dt_recall:.4f}",
        f"{dt_f1:.4f}",
        f"{dt_rmse:.4f}",
        f"{dt_mae:.4f}",
        f"{dt_training_time:.2f}"
    ]
})

print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✓ Results saved to: model_comparison_results.csv")

# ============================================================================
# 8. VISUALIZATION OF RESULTS
# ============================================================================

print("\n" + "="*70)
print("[STEP 7] CREATING RESULT VISUALIZATIONS")
print("="*70)

# Create comprehensive comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Comparison: KNN vs Decision Tree Regressor', 
             fontsize=16, fontweight='bold')

# Plot 1: Bar chart comparison - Classification Metrics
ax1 = axes[0, 0]
metrics_class = ['Precision', 'Recall', 'F1-Score']
knn_scores = [knn_precision, knn_recall, knn_f1]
dt_scores = [dt_precision, dt_recall, dt_f1]

x = np.arange(len(metrics_class))
width = 0.35

bars1 = ax1.bar(x - width/2, knn_scores, width, label='KNN', color='skyblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, dt_scores, width, label='Decision Tree', color='lightcoral', edgecolor='black')

ax1.set_xlabel('Metrics', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Classification Metrics Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_class)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 2: Bar chart comparison - Regression Metrics
ax2 = axes[0, 1]
metrics_reg = ['RMSE', 'MAE']
knn_reg = [knn_rmse, knn_mae]
dt_reg = [dt_rmse, dt_mae]

x = np.arange(len(metrics_reg))
bars1 = ax2.bar(x - width/2, knn_reg, width, label='KNN', color='lightgreen', edgecolor='black')
bars2 = ax2.bar(x + width/2, dt_reg, width, label='Decision Tree', color='plum', edgecolor='black')

ax2.set_xlabel('Metrics', fontweight='bold')
ax2.set_ylabel('Error', fontweight='bold')
ax2.set_title('Regression Metrics Comparison (Lower is Better)')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_reg)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Training Time Comparison
ax3 = axes[0, 2]
times = [knn_training_time, dt_training_time]
colors_time = ['skyblue', 'lightcoral']
bars = ax3.bar(['KNN', 'Decision Tree'], times, color=colors_time, edgecolor='black')
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Training Time Comparison')
ax3.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}s', ha='center', va='bottom', fontweight='bold')

# Plot 4: Prediction Error Distribution - KNN
ax4 = axes[1, 0]
knn_errors = y_test.values - knn_predictions
ax4.hist(knn_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Prediction Error', fontweight='bold')
ax4.set_ylabel('Frequency', fontweight='bold')
ax4.set_title('KNN - Prediction Error Distribution')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Prediction Error Distribution - Decision Tree
ax5 = axes[1, 1]
dt_errors = y_test.values - dt_predictions
ax5.hist(dt_errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('Prediction Error', fontweight='bold')
ax5.set_ylabel('Frequency', fontweight='bold')
ax5.set_title('Decision Tree - Prediction Error Distribution')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# Plot 6: Confusion Matrix Comparison
ax6 = axes[1, 2]
knn_cm = confusion_matrix(y_test_binary, knn_pred_binary)
dt_cm = confusion_matrix(y_test_binary, dt_pred_binary)

# Create side-by-side confusion matrices
cm_comparison = np.hstack([knn_cm, [[0, 0], [0, 0]], dt_cm])
im = ax6.imshow(cm_comparison, cmap='Blues', aspect='auto')

ax6.set_xticks([0.5, 2.5, 5.5])
ax6.set_xticklabels(['KNN', '', 'Decision Tree'])
ax6.set_yticks([0, 1])
ax6.set_yticklabels(['Not Relevant', 'Relevant'])
ax6.set_title('Confusion Matrix Comparison')

# Add text annotations
for i in range(2):
    for j in range(2):
        ax6.text(j, i, str(knn_cm[i, j]), ha='center', va='center', color='black', fontweight='bold')
        ax6.text(j+4, i, str(dt_cm[i, j]), ha='center', va='center', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: model_comparison_results.png")
plt.show()

# ============================================================================
# 9. ADDITIONAL ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("[STEP 8] ADDITIONAL ANALYSIS")
print("="*70)

# 9.1 Learning Curve Analysis
print("\n[9.1] LEARNING CURVE ANALYSIS...")
print("-"*70)

train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
knn_train_scores = []
knn_test_scores = []
dt_train_scores = []
dt_test_scores = []

print("Computing learning curves...")
for size in train_sizes:
    # Sample data
    sample_size = int(len(X_train) * size)
    X_sample = X_train_scaled[:sample_size]
    y_sample = y_train.iloc[:sample_size]
    
    # Train and evaluate KNN
    best_knn.fit(X_sample, y_sample)
    knn_train_pred = best_knn.predict(X_sample)
    knn_test_pred = best_knn.predict(X_test_scaled)
    
    knn_train_rmse = np.sqrt(mean_squared_error(y_sample, knn_train_pred))
    knn_test_rmse = np.sqrt(mean_squared_error(y_test, knn_test_pred))
    
    knn_train_scores.append(knn_train_rmse)
    knn_test_scores.append(knn_test_rmse)
    
    # Train and evaluate Decision Tree
    best_dt.fit(X_sample, y_sample)
    dt_train_pred = best_dt.predict(X_sample)
    dt_test_pred = best_dt.predict(X_test_scaled)
    
    dt_train_rmse = np.sqrt(mean_squared_error(y_sample, dt_train_pred))
    dt_test_rmse = np.sqrt(mean_squared_error(y_test, dt_test_pred))
    
    dt_train_scores.append(dt_train_rmse)
    dt_test_scores.append(dt_test_rmse)
    
    print(f"  Size: {size*100:.0f}% - KNN Test RMSE: {knn_test_rmse:.4f}, DT Test RMSE: {dt_test_rmse:.4f}")

# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Learning Curves: Training Size vs RMSE', fontsize=14, fontweight='bold')

# KNN Learning Curve
ax1 = axes[0]
ax1.plot([s*100 for s in train_sizes], knn_train_scores, 'o-', color='blue', 
         label='Training RMSE', linewidth=2, markersize=8)
ax1.plot([s*100 for s in train_sizes], knn_test_scores, 's-', color='red', 
         label='Testing RMSE', linewidth=2, markersize=8)
ax1.set_xlabel('Training Size (%)', fontweight='bold')
ax1.set_ylabel('RMSE', fontweight='bold')
ax1.set_title('K-Nearest Neighbors (KNN)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Decision Tree Learning Curve
ax2 = axes[1]
ax2.plot([s*100 for s in train_sizes], dt_train_scores, 'o-', color='blue', 
         label='Training RMSE', linewidth=2, markersize=8)
ax2.plot([s*100 for s in train_sizes], dt_test_scores, 's-', color='red', 
         label='Testing RMSE', linewidth=2, markersize=8)
ax2.set_xlabel('Training Size (%)', fontweight='bold')
ax2.set_ylabel('RMSE', fontweight='bold')
ax2.set_title('Decision Tree Regressor')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ Learning curve saved: learning_curves.png")
plt.show()

# 9.2 Feature Importance (Decision Tree only)
print("\n[9.2] FEATURE IMPORTANCE ANALYSIS (Decision Tree)")
print("-"*70)

# Get feature importances
feature_importance = best_dt.feature_importances_
feature_names = X.columns

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n[Top 10 Most Important Features]")
print(importance_df.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 6))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'], color='teal', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance Score', fontweight='bold')
plt.ylabel('Features', fontweight='bold')
plt.title('Top 15 Feature Importances (Decision Tree Regressor)', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved: feature_importance.png")
plt.show()

# 9.3 Prediction Distribution Analysis
print("\n[9.3] PREDICTION DISTRIBUTION ANALYSIS")
print("-"*70)

# Create prediction distribution plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Prediction Distribution Analysis', fontsize=14, fontweight='bold')

# Actual ratings distribution
ax1 = axes[0]
ax1.hist(y_test, bins=5, color='gray', edgecolor='black', alpha=0.7, label='Actual')
ax1.set_xlabel('Rating', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Actual Ratings Distribution')
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# KNN predictions distribution
ax2 = axes[1]
ax2.hist(knn_predictions, bins=20, color='skyblue', edgecolor='black', alpha=0.7, label='KNN Predictions')
ax2.set_xlabel('Rating', fontweight='bold')
ax2.set_ylabel('Frequency', fontweight='bold')
ax2.set_title('KNN Predictions Distribution')
ax2.axvline(knn_predictions.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {knn_predictions.mean():.2f}')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# Decision Tree predictions distribution
ax3 = axes[2]
ax3.hist(dt_predictions, bins=20, color='lightcoral', edgecolor='black', alpha=0.7, label='DT Predictions')
ax3.set_xlabel('Rating', fontweight='bold')
ax3.set_ylabel('Frequency', fontweight='bold')
ax3.set_title('Decision Tree Predictions Distribution')
ax3.axvline(dt_predictions.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: {dt_predictions.mean():.2f}')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Prediction distribution plot saved: prediction_distribution.png")
plt.show()

# 9.4 Scatter Plot: Actual vs Predicted
print("\n[9.4] ACTUAL VS PREDICTED SCATTER PLOT")
print("-"*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Actual vs Predicted Ratings', fontsize=14, fontweight='bold')

# Sample data for better visualization
sample_size = min(1000, len(y_test))
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test.iloc[sample_indices]
knn_pred_sample = knn_predictions[sample_indices]
dt_pred_sample = dt_predictions[sample_indices]

# KNN scatter plot
ax1 = axes[0]
ax1.scatter(y_test_sample, knn_pred_sample, alpha=0.5, c='skyblue', edgecolors='black', s=30)
ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Rating', fontweight='bold')
ax1.set_ylabel('Predicted Rating', fontweight='bold')
ax1.set_title(f'KNN (RMSE: {knn_rmse:.4f})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.5, 5.5])
ax1.set_ylim([0.5, 5.5])

# Decision Tree scatter plot
ax2 = axes[1]
ax2.scatter(y_test_sample, dt_pred_sample, alpha=0.5, c='lightcoral', edgecolors='black', s=30)
ax2.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Rating', fontweight='bold')
ax2.set_ylabel('Predicted Rating', fontweight='bold')
ax2.set_title(f'Decision Tree (RMSE: {dt_rmse:.4f})')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.5, 5.5])
ax2.set_ylim([0.5, 5.5])

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("\n✓ Scatter plot saved: actual_vs_predicted.png")
plt.show()

# ============================================================================
# 10. EXAMPLE RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("[STEP 9] GENERATE SAMPLE RECOMMENDATIONS")
print("="*70)

# 10.1 Function to get movie recommendations
def get_movie_recommendations(user_id, model, X_data, n_recommendations=10):
    """
    Generate movie recommendations for a given user
    """
    # Get movies the user hasn't rated
    user_ratings = data[data['user_id'] == user_id]
    rated_movies = set(user_ratings['movie_id'].unique())
    all_movies = set(data['movie_id'].unique())
    unrated_movies = list(all_movies - rated_movies)
    
    # Create feature matrix for unrated movies
    user_features = X_data[X_data.index.isin(data[data['user_id'] == user_id].index)].iloc[0:1]
    
    recommendations = []
    for movie_id in unrated_movies[:100]:  # Limit to 100 for speed
        # Get movie features from training data
        movie_data = data[data['movie_id'] == movie_id].iloc[0:1]
        
        if len(movie_data) > 0:
            # Create feature vector
            feature_vector = user_features.copy()
            feature_vector['movie_id'] = movie_id
            
            # Get movie info
            movie_info = movies[movies['movie_id'] == movie_id]
            if len(movie_info) > 0:
                # Predict rating
                pred_rating = model.predict(feature_vector.values)[0]
                pred_rating = np.clip(pred_rating, 1, 5)
                
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_info['title'].values[0],
                    'predicted_rating': pred_rating
                })
    
    # Sort by predicted rating
    recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
    return recommendations[:n_recommendations]

# 10.2 Generate sample recommendations
print("\n[10.1] GENERATING SAMPLE RECOMMENDATIONS...")
print("-"*70)

sample_user_id = 1
print(f"\nGenerating recommendations for User ID: {sample_user_id}")

# Get user's actual ratings
user_actual_ratings = data[data['user_id'] == sample_user_id][['movie_id', 'title', 'rating']].head(10)
print(f"\n[User {sample_user_id}'s Actual Ratings (Sample)]")
print(user_actual_ratings.to_string(index=False))

# Note: For simplicity in this demo, we'll show top predicted movies from test set
print("\n[Sample Top Predicted Movies - KNN]")
sample_test_indices = X_test[X_test['user_id'] == sample_user_id].index[:5]
if len(sample_test_indices) > 0:
    sample_movies = data.loc[sample_test_indices, ['title', 'rating']]
    sample_predictions = pd.DataFrame({
        'Movie Title': sample_movies['title'].values,
        'Actual Rating': sample_movies['rating'].values,
        'KNN Predicted': knn_predictions[X_test.index.isin(sample_test_indices)][:len(sample_movies)],
        'DT Predicted': dt_predictions[X_test.index.isin(sample_test_indices)][:len(sample_movies)]
    })
    print(sample_predictions.to_string(index=False))
else:
    print("  No test samples available for this user")

# ============================================================================
# 11. SAVE MODELS
# ============================================================================

print("\n" + "="*70)
print("[STEP 10] SAVING MODELS")
print("="*70)

import pickle

# Save models
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)
print("✓ KNN model saved: knn_model.pkl")

with open('dt_model.pkl', 'wb') as f:
    pickle.dump(best_dt, f)
print("✓ Decision Tree model saved: dt_model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

# Save encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'gender': le_gender, 'occupation': le_occupation}, f)
print("✓ Label encoders saved: label_encoders.pkl")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[FINAL SUMMARY]")
print("="*70)

summary_text = f"""
EXPERIMENT COMPLETED SUCCESSFULLY!

Dataset Information:
- Total Ratings: {len(ratings):,}
- Unique Users: {ratings['user_id'].nunique():,}
- Unique Movies: {ratings['movie_id'].nunique():,}
- Sparsity: {sparsity:.2f}%

Data Split:
- Training Set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)
- Testing Set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)

K-NEAREST NEIGHBORS (KNN):
- Best Parameters: {grid_knn.best_params_}
- Training Time: {knn_training_time:.2f} seconds
- RMSE: {knn_rmse:.4f}
- MAE: {knn_mae:.4f}
- Precision: {knn_precision:.4f}
- Recall: {knn_recall:.4f}
- F1-Score: {knn_f1:.4f}

DECISION TREE REGRESSOR:
- Best Parameters: {grid_dt.best_params_}
- Training Time: {dt_training_time:.2f} seconds
- RMSE: {dt_rmse:.4f}
- MAE: {dt_mae:.4f}
- Precision: {dt_precision:.4f}
- Recall: {dt_recall:.4f}
- F1-Score: {dt_f1:.4f}

WINNER (Based on RMSE):
{'KNN' if knn_rmse < dt_rmse else 'Decision Tree Regressor'} performs better with RMSE of {min(knn_rmse, dt_rmse):.4f}

Output Files Generated:
1. eda_analysis.png - Exploratory Data Analysis visualizations
2. model_comparison_results.png - Model performance comparison
3. model_comparison_results.csv - Metrics comparison table
4. learning_curves.png - Learning curve analysis
5. feature_importance.png - Feature importance (Decision Tree)
6. prediction_distribution.png - Prediction distributions
7. actual_vs_predicted.png - Scatter plots
8. knn_model.pkl - Trained KNN model
9. dt_model.pkl - Trained Decision Tree model
10. scaler.pkl - Feature scaler
11. label_encoders.pkl - Label encoders

Experiment Duration: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary_text)

# Save summary to text file
with open('experiment_summary.txt', 'w') as f:
    f.write(summary_text)
print("✓ Summary saved: experiment_summary.txt")

print("\n" + "="*70)
print("EXPERIMENT COMPLETED!")
print("="*70)
print("\nYou can now use these results for your BAB IV (Results and Discussion)")
print("All visualizations and metrics have been saved.")
print("\nNext Steps:")
print("1. Review all generated PNG files")
print("2. Analyze the comparison table (CSV)")
print("3. Write BAB IV based on these results")
print("4. Write BAB V (Conclusions and Recommendations)")
print("="*70)