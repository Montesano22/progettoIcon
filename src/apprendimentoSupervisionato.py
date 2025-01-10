import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from scipy.stats import gmean
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.svm import LinearSVC
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class SklearnCompatibleXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y, **fit_params):
        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

# =====================
# 1) CARICAMENTO DATI
# =====================
df = pd.read_csv('../datasets/cleaned_olympics_dataset.csv')

# =====================
# 2) PULIZIA DEI DATI
# =====================
df['Medal'].fillna('None', inplace=True)
df.dropna(subset=['Sex', 'Age', 'Height', 'Weight', 'Team', 'Year', 'Sport', 'Event', 'Medal'], inplace=True)

# =====================
# 3) FUTURE ENGINEERING
# =====================

df['Medal'] = df['Medal'].apply(lambda x: 0 if x == 'None' else 1)

# 1. Calcolo del numero di medaglie per Team
df['Team_Medals'] = df.groupby('Team')['Medal'].transform('sum')

# 2. Probabilità storiche di medaglia
df['Sport_Medal_Probability'] = df.groupby('Sport')['Medal'].transform('mean')
df['Team_Medal_Probability'] = df.groupby('Team')['Medal'].transform('mean')

# 3. Rapporto Altezza/Peso (normalizzato)
df['Height_to_Weight_Ratio'] = df['Height'] / df['Weight']
df['Height_to_Weight_Ratio'] = (df['Height_to_Weight_Ratio'] - df['Height_to_Weight_Ratio'].mean()) / df['Height_to_Weight_Ratio'].std()

# 4. Distanza dall'Anno Corrente
df['Years_Since_Event'] = 2016 - df['Year']  # Sostituire 2016 con l'anno corrente, se necessario

# 5. Età come Categorie
bins = [0, 18, 25, 35, 50, 100]
labels = ['Junior', 'Under 25', 'Prime', 'Veteran', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
df['Age_Group'] = df['Age_Group'].cat.codes  # Trasforma in numerico

# 6. Competizione Internazionale vs Nazionale
df['Is_Host_Country'] = (df['Team'] == 'Host_Country').astype(int)

# 7. Importanza dello Sport
df['Sport_Importance'] = df.groupby('Sport')['Medal'].transform('mean')

# 8. Altezza relativa per Sport
df['Height_Relative'] = df['Height'] / df.groupby('Sport')['Height'].transform('mean')

# 9. Numero di Atleti per Evento
df['Event_Competitor_Count'] = df.groupby(['Event', 'Year'])['Name'].transform('count')

# 10. Competitività dello Sport (usando Event_Competitor_Count)
df['Sport_Competitiveness'] = 1 / df.groupby('Sport')['Event_Competitor_Count'].transform('mean')  # Competitività inversa

# 11. Competitività dell'Evento
df['Event_Competitiveness'] = 1 / df['Event_Competitor_Count']

# 12. Performance di Squadra
df['Team_Avg_Medals_Year'] = df.groupby(['Team', 'Year'])['Medal'].transform('mean')

# 13. Probabilità Individuale Basata su Eventi
df['Event_Medal_Probability'] = df.groupby('Event')['Medal'].transform('mean')

# 14. Paese come Leader nel Medagliere
top_countries = ['USA', 'CHN', 'RUS']
df['Is_Top_Country'] = df['Team'].apply(lambda x: 1 if x in top_countries else 0)

# 15. Combinazioni di Età e Sport
df['Sport_Age_Alignment'] = df['Age'] / df.groupby('Sport')['Age'].transform('mean')

# 16. Presenza in Eventi Multipli
df['Events_Per_Year'] = df.groupby(['Name', 'Year'])['Event'].transform('count')

# 17. Esperienza Revisata
df['First_Year'] = df.groupby('Name')['Year'].transform('min')
df['Experience_Years'] = df['Year'] - df['First_Year']
df['Experience_Ratio'] = df['Experience_Years'] / df['Age']  # Rapporto esperienza/età

# 18. Performance Passate
df['Previous_Medals'] = df.groupby('Name')['Medal'].cumsum() - df['Medal']
df['Medals_Per_Participation'] = df['Previous_Medals'] / df['Events_Per_Year'].replace(0, np.nan)  # Medaglie per evento

# =====================
# Creazione del Ranking Combinato
# =====================
df['Ranking_Proxy'] = (
    0.3 * df['Team_Medals'].rank(ascending=False, method='dense').fillna(0) +
    0.2 * df['Sport_Medal_Probability'].rank(ascending=False, method='dense').fillna(0) +
    0.2 * df['Team_Medal_Probability'].rank(ascending=False, method='dense').fillna(0) +
    0.2 * (1 / (df['Weight'] + 1)).rank(ascending=False, method='dense') +
    0.1 * df['Age'].rank(ascending=True, method='dense')
)

# Normalizziamo il ranking
if 'Ranking_Proxy' in df.columns:
    df['Ranking_Proxy'] = (df['Ranking_Proxy'] - df['Ranking_Proxy'].min()) / (df['Ranking_Proxy'].max() - df['Ranking_Proxy'].min())

# =====================
# Gestione delle Feature Altamente Correlate
# =====================
correlation_threshold = 0.9
numerical_columns = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_columns.corr().abs()

# Identifica le feature altamente correlate
correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > correlation_threshold:
            correlated_features.add(correlation_matrix.columns[i])

print(f"Feature altamente correlate rimosse: {correlated_features}")
df.drop(columns=correlated_features, inplace=True, errors='ignore')


# ===============================
# 4) ANALISI DELLA CORRELAZIONE
# ===============================
# Selezioniamo solo le colonne numeriche
numerical_columns = df.select_dtypes(include=[np.number])
correlation_matrix = numerical_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Heatmap della Correlazione")
plt.show()

# Soglia per rimuovere le feature altamente correlate
correlation_threshold = 0.9
correlated_features = set()
correlation_matrix = correlation_matrix.abs()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > correlation_threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

print(f"Feature altamente correlate rimosse: {correlated_features}")
df = df.drop(columns=correlated_features, errors='ignore')

# =====================
# 5) CALCOLO DELL'IMPORTANZA DELLE FEATURE
# =====================
categorical_cols = ['Sex', 'Team', 'Sport', 'Event']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =====================
# Selezione delle feature per X e y
# =====================
# Controlla quali colonne sono ancora presenti nel DataFrame
available_columns = set(df.columns)

# Seleziona solo le colonne disponibili
selected_features = [
    'Sex', 'Age', 'Height', 'Weight', 'Team', 'Year', 'Sport', 'Event',
    'Sport_Medal_Probability', 'Team_Medal_Probability',
    'Ranking_Proxy', 'Num_Participations', 'Height_Relative',
    'Team_Avg_Medals_Year', 'Event_Competitor_Count',
    'Event_Medal_Probability', 'Is_Top_Country', 'Sport_Age_Alignment',
    'Events_Per_Year', 'Experience_Years', 'Previous_Medals'
]

# Filtro sulle feature che esistono ancora
final_features = [feature for feature in selected_features if feature in available_columns]

# Definizione di X e y
X = df[final_features]
y = df['Medal']

# Calcolo dell'importanza delle feature con Random Forest
temp_model = RandomForestClassifier(random_state=42)
temp_model.fit(X, y)
importances = temp_model.feature_importances_

# Ordiniamo le feature in base all'importanza
importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("Importanza delle Feature:")
print(importance_df)

# Rimuoviamo feature con importanza molto bassa (es: sotto 0.01)
low_importance_features = importance_df[importance_df["Importance"] < 0.01]["Feature"].tolist()
print(f"Feature a bassa importanza rimosse: {low_importance_features}")
X = X.drop(columns=low_importance_features)

# =====================
# 6) RIMOZIONE DEGLI OUTLIER
# =====================
def remove_outliers_with_log(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Outliers rimossi per {col}: {len(outliers)}")
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

numeric_columns = ['Age', 'Height', 'Weight']
df = remove_outliers_with_log(df, numeric_columns)  # Usa il nome corretto della funzione
print("Dati dopo la rimozione degli outlier:", df.shape)

# =====================
# 7) TRAIN/TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================
# 8) NORMALIZZAZIONE
# =====================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================
# 9) COMBINAZIONE DI OVERSAMPLING E UNDERSAMPLING (SMOTE-Tomek)
# =====================
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
print("Distribuzione dopo SMOTE-Tomek:", np.unique(y_train_balanced, return_counts=True))

# =====================
# 10) FUNZIONI DI SUPPORTO
# =====================
def plot_learning_curve(model, model_name, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)

    plt.title(f'Learning Curve: {model_name}')
    plt.xlabel('Training size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def train_and_evaluate_model_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search per trovare i migliori parametri
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv_method,
        scoring='f1',
        n_jobs=-1,
        error_score=np.nan  # Ignora i fold che falliscono
    )
    grid_search.fit(X_train, y_train)

    # Miglior modello trovato
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\n--- {model_name} ---")
    print("Best params:", grid_search.best_params_)
    print(classification_report(y_test, y_pred, target_names=["No Medal", "Medal"]))

    # Cross-validation results
    cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=cv_method, scoring='accuracy')
    print("\nRisultati della Cross-Validation:")
    print(f"Cross-Validation Accuracy (mean ± std): {np.mean(cross_val_scores):.4f} ± {np.std(cross_val_scores):.4f}")

    # Accuracy su training e test
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Geometric Mean Average Precision (GMAP)
    if np.all(cross_val_scores > 0):  # Controlla che tutti i valori siano positivi
        gmap = gmean(cross_val_scores)
        print(f"Geometric Mean Average Precision (GMAP): {gmap:.4f} (± {np.std(cross_val_scores):.4f})")
    else:
        print("GMAP non calcolabile a causa di valori non validi.")

    # Curva di apprendimento (richiede una funzione separata)
    try:
        plot_learning_curve(best_model, model_name, X_train, y_train)
    except NameError:
        print("Funzione 'plot_learning_curve' non definita. Verifica la sua implementazione.")

def main():
    while True:
        print("\nModelli di Machine Learning:")
        print("1. Random Forest (balanced)")
        print("2. Support Vector Machine")
        print("3. Gradient Boosting (XGBoost)")
        print("0. Exit")
        choice = input("Inserisci il numero del modello scelto: ")

        if choice == '1':
            # Balanced Random Forest
            print("Tempo stimato: tra i 10 - 15 minuti")
            brf_param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            brf_model = BalancedRandomForestClassifier(random_state=42)
            train_and_evaluate_model_grid_search(
                brf_model, brf_param_grid,
                X_train_balanced, y_train_balanced, X_test, y_test,
                "Random Forest"
            )

        elif choice == '2':
            print("Tempo necessario: tra i 1 - 2 minuti")

            # Griglia di parametri per LinearSVC
            svc_param_grid = {
                'C': [0.1, 1, 10],  # Parametro di regolarizzazione
                'class_weight': ['balanced'],  # Gestione dello sbilanciamento delle classi
                'max_iter': [2000]  # Iterazioni massime
            }

            # Modello LinearSVC
            svc_model = LinearSVC(random_state=42, dual=False)  # dual=False ottimizza per piccoli dataset

            # Funzione di train e valutazione con grid search
            train_and_evaluate_model_grid_search(
                svc_model, svc_param_grid,
                X_train, y_train, X_test, y_test,  # Usa i dati originali
                "Support Vector Machine (LinearSVC)"
            )

        elif choice == '3':
            print("Tempo stimato: tra 1 e 2 minuti")

            # Griglia di parametri per GridSearchCV
            xgb_param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
            }

            # Calcolo del peso per la classe positiva
            scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])

            # Modello XGBoost con bilanciamento
            xgb_model = SklearnCompatibleXGBClassifier(
                objective='binary:logistic',  # Per problemi binari
                random_state=42,
                scale_pos_weight=scale_pos_weight  # Peso bilanciato per le classi
            )

            # Funzione di training e valutazione
            train_and_evaluate_model_grid_search(
                xgb_model, xgb_param_grid,
                X_train_balanced, y_train_balanced, X_test, y_test,
                "Gradient Boosting Classifier"
            )

        elif choice == '0':
            print("Uscita...")
            break

        else:
            print("Scelta non valida. Riprova.")


# ==========================
# Avvio del programma
# ==========================
if __name__ == "__main__":
    main()
