import pandas as pd
# Carica il dataset
file_path = 'dataset_olympics.csv'  # Sostituisci con il percorso corretto
olympics_data = pd.read_csv(file_path)

# 1. Rimuovere righe duplicate
olympics_data = olympics_data.drop_duplicates()

# 2. Gestire valori mancanti
# Sostituire i valori mancanti in colonne categoriali con un'etichetta "Unknown"
olympics_data['Medal'] = olympics_data['Medal'].fillna('None')

# Rimuovere righe con valori mancanti critici
critical_columns = ['Name', 'Sport', 'Team', 'Year', 'Event']
olympics_data = olympics_data.dropna(subset=critical_columns)

# Per le colonne numeriche (es. Height, Weight), sostituire i valori mancanti con la mediana
for col in ['Age', 'Height', 'Weight']:
    olympics_data[col] = olympics_data[col].fillna(olympics_data[col].median())

# 3. Uniformare i valori delle colonne
# Convertire le stringhe in minuscolo per uniformità
olympics_data['Name'] = olympics_data['Name'].str.title()
olympics_data['Sport'] = olympics_data['Sport'].str.title()
olympics_data['Team'] = olympics_data['Team'].str.title()
olympics_data['Event'] = olympics_data['Event'].str.title()

# 4. Rimuovere colonne non necessarie per la KB
kb_columns = ['Name', 'Sex', 'Age', 'Height', 'Weight', 'Team', 'Year', 'Sport', 'Event', 'Medal']
kb_ready_data = olympics_data[kb_columns]

# 5. Salva il dataset pulito
output_path = 'cleaned_olympics_dataset.csv'
kb_ready_data.to_csv(output_path, index=False)

print(f"Dataset pulito salvato in: {output_path}")
