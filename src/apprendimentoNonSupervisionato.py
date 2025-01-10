import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

def main():
    # Configurazione di stile
    sns.set(style="whitegrid")

    # Caricare il file CSV
    data_path = '../datasets/cleaned_olympics_dataset.csv'
    print("[INFO] Caricamento del file CSV...")
    df = pd.read_csv(data_path)
    print("[INFO] File caricato con successo.")

    # Mostrare le prime righe per capire la struttura
    print("[INFO] Prime righe del dataset:")
    print(df.head())

    # Preprocessing: selezionare solo colonne numeriche
    print("[INFO] Selezione delle colonne numeriche...")
    numeric_df = df[['Age', 'Height', 'Weight']].dropna()
    print(f"[INFO] Dimensioni del dataset numerico: {numeric_df.shape}")

    # Standardizzare i dati
    print("[INFO] Standardizzazione dei dati...")
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df)
    print("[INFO] Standardizzazione completata.")

    # Riduzione della dimensionalità
    print("[INFO] Riduzione della dimensionalità dei dati...")
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(f"[INFO] Dimensioni ridotte del dataset: {X_reduced.shape}")

    # Calcolare i loadings del PCA
    pca_loadings = pd.DataFrame(pca.components_, columns=['Age', 'Height', 'Weight'], index=['Componente 1', 'Componente 2'])
    print("[INFO] Loadings del PCA (contributo delle variabili originali):")
    print(pca_loadings)

    # Metodo del gomito
    print("[INFO] Metodo del Gomito...")
    inertia = []
    k_values = range(2, 11)
    for k in k_values:
        print(f"[INFO] Esecuzione KMeans per k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_reduced)
        inertia.append(kmeans.inertia_)

    # Visualizzare il grafico del gomito
    print("[INFO] Metodo del Gomito completato. Visualizzazione del grafico...")
    plt.figure(figsize=(10, 6))
    plt.style.use('default')
    plt.plot(k_values, inertia, marker='o', linestyle='--', color='royalblue')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Metodo del Gomito per Determinare k Ottimale')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calcolo Silhouette Score solo per k=3 con interruzione Ctrl+C
    try:
        print("[INFO] Calcolo Silhouette Score per k=3... CTRL+C per saltare")
        time.sleep(1)  # Pausa per evitare stampa immediata
        k = 3
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_reduced)
        silhouette_avg = silhouette_score(X_reduced, labels)
        print(f"[INFO] Silhouette Score per k=3: {silhouette_avg:.4f}")
    except KeyboardInterrupt:
        print("\n[INFO] Calcolo interrotto manualmente con CTRL+C.")
        silhouette_avg = None

    # Etichettare i cluster nel DataFrame originale
    numeric_df['Cluster'] = labels
    df = df.dropna(subset=['Age', 'Height', 'Weight'])  # Allinea righe con numeric_df
    df['Cluster'] = labels  # Propaga la colonna 'Cluster' al DataFrame originale

    # Analisi dei cluster
    print("[INFO] Analisi delle caratteristiche dei cluster...")
    cluster_summary = numeric_df.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Height': ['mean', 'std'],
        'Weight': ['mean', 'std']
    }).round(2)

    print("[INFO] Statistiche riassuntive per cluster:")
    print(cluster_summary)

    # Grafici per la distribuzione degli sport nei cluster
    if 'Sport' in df.columns:
        for cluster in df['Cluster'].unique():
            plt.figure(figsize=(16, 8))
            sport_distribution = df[df['Cluster'] == cluster]['Sport'].value_counts()
            sport_distribution.plot(kind='bar', color='skyblue')
            cluster_info = (
                f"Cluster {cluster}:\n"
                f"Età media = {cluster_summary.loc[cluster, ('Age', 'mean')]:.2f}\n"
                f"Altezza media = {cluster_summary.loc[cluster, ('Height', 'mean')]:.2f} cm\n"
                f"Peso medio = {cluster_summary.loc[cluster, ('Weight', 'mean')]:.2f} kg"
            )
            plt.title(f"Distribuzione degli Sport nel Cluster {cluster}\n{cluster_info}", fontsize=16)
            plt.xlabel("Sport", fontsize=12)
            plt.ylabel("Conteggio", fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.tight_layout()
            plt.show()

    # Scatterplot con annotazioni delle caratteristiche principali dei cluster
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", 3)
    for i in range(3):
        cluster_text = (
            f"Cluster {i}\n"
            f"Età media: {cluster_summary.loc[i, ('Age', 'mean')]:.2f} ± {cluster_summary.loc[i, ('Age', 'std')]:.2f}\n"
            f"Altezza media: {cluster_summary.loc[i, ('Height', 'mean')]:.2f} ± {cluster_summary.loc[i, ('Height', 'std')]:.2f} cm\n"
            f"Peso medio: {cluster_summary.loc[i, ('Weight', 'mean')]:.2f} ± {cluster_summary.loc[i, ('Weight', 'std')]:.2f} kg"
        )
        plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=cluster_text, alpha=0.7, edgecolors='k', s=70, color=palette[i])

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=250, label='Centroidi', edgecolor='black')
    plt.xlabel('Dimensione Principale 1 (Combinazione di Età, Altezza, Peso)', fontsize=12)
    plt.ylabel('Dimensione Principale 2 (Seconda Combinazione di Età, Altezza, Peso)', fontsize=12)
    plt.title("Cluster degli Atleti Basati su Caratteristiche Fisiche (KMeans + PCA)", fontsize=14)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper right', frameon=True, fontsize=10)  # Posizionamento in alto a destra
    plt.tight_layout()
    plt.show()

    # Identificazione delle anomalie con Isolation Forest
    print("[INFO] Identificazione delle anomalie con Isolation Forest...")
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    anomaly_labels = isolation_forest.fit_predict(X)

    # Separare le anomalie dai dati normali
    df['Anomaly'] = anomaly_labels
    anomalies = df[df['Anomaly'] == -1]
    normal_data = df[df['Anomaly'] == 1]
    print(f"[INFO] Numero di anomalie rilevate: {len(anomalies)}")

    # Visualizzare i cluster con le anomalie
    print("[INFO] Visualizzazione dei cluster con anomalie...")
    plt.figure(figsize=(12, 8))
    for i in range(3):
        plt.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], label=f"Cluster {i}", alpha=0.5, edgecolors='k', s=70, color=palette[i])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=250, label='Centroidi', edgecolor='black')
    plt.scatter(X_reduced[anomaly_labels == -1, 0], X_reduced[anomaly_labels == -1, 1], c='orange', marker='o', s=100, edgecolor='black', label='Anomalie')
    plt.title("Visualizzazione dei Cluster con Anomalie", fontsize=14)
    plt.xlabel('Componente Principale 1', fontsize=12)
    plt.ylabel('Componente Principale 2', fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.show()

    # Etichettare i cluster e anomalie nel DataFrame originale
    df['Cluster'] = labels
    df['Anomaly'] = anomaly_labels
    print("[INFO] Esempio di dati con etichetta del cluster e anomalie assegnate:")
    print(df.head())

if __name__ == "__main__":
    main()


