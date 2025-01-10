import pandas as pd

# Percorso del dataset corretto
cleaned_dataset_path = "cleaned_olympics_dataset.csv"
output_prolog_file = "olympics_kb_fixed_v2.pl"

# Carica il dataset
data = pd.read_csv(cleaned_dataset_path)

# Funzione per pulire e formattare le stringhe per Prolog
def clean_string(value):
    if pd.isna(value):
        return "none"  # Rimpiazza valori NaN con 'none'
    return str(value).replace("'", "\\'")  # Escape per apostrofi singoli

# Genera il file Prolog
with open(output_prolog_file, "w", encoding="utf-8") as file:
    file.write("%% Knowledge Base degli atleti olimpici\n\n")
    for _, row in data.iterrows():
        try:
            fact = (
                f"atleta('{clean_string(row['Name'])}', '{clean_string(row['Sex'])}', "
                f"{int(row['Age']) if not pd.isna(row['Age']) else 'none'}, "
                f"{int(row['Height']) if not pd.isna(row['Height']) else 'none'}, "
                f"{int(row['Weight']) if not pd.isna(row['Weight']) else 'none'}, "
                f"'{clean_string(row['Team'])}', {int(row['Year'])}, "
                f"'{clean_string(row['Sport'])}', '{clean_string(row['Event'])}', "
                f"'{clean_string(row['Medal'])}').\n"
            )
            file.write(fact)
        except Exception as e:
            print(f"Errore con riga: {row}\nErrore: {e}")

print(f"File Prolog generato correttamente: {output_prolog_file}")
