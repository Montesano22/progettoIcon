from pyswip import Prolog
from tabulate import tabulate

# Inizializza SWI-Prolog e carica la Knowledge Base
prolog = Prolog()

try:
    prolog.consult("olympicskb.pl")
except Exception as e:
    print(f"Errore durante il caricamento della Knowledge Base: {e}")
    exit()

def format_value(value, max_length=50):
    """Formatta i valori per evitare problemi di visualizzazione."""
    if isinstance(value, str):
        value = value.strip().replace('"', '').replace("'", "").replace("(", "").replace(")", "")
        return value[:max_length] + "..." if len(value) > max_length else value
    return value

def format_row(row, max_length=50):
    """Applica formattazione a ogni valore in una riga."""
    return [format_value(value, max_length) for value in row]

def main():
    while True:
        try:
            print("\n--- Menu Opzioni ---")
            print("1. Cerca informazioni su un atleta")
            print("2. Trova tutti gli atleti di una nazionalità")
            print("3. Trova atleti che hanno vinto medaglie in uno sport")
            print("4. Trova il miglior risultato di un atleta in termini di medaglia")
            print("5. Trova gli sport più popolari in termini di medaglie vinte da una nazione")
            print("6. Trova l'atleta più giovane/anziano che ha vinto una medaglia")
            print("7. Trova il rapporto medaglie/partecipanti per una nazione")
            print("8. Trova la distribuzione delle medaglie per edizione")
            print("9. Esci")

            scelta = input("Scegli un'opzione (1-9): ")

            if scelta == "1":
                nome = input("Inserisci il nome o parte del nome dell'atleta: ").strip().title()
                query = f"cerca_atleta_parziale('{nome}', Nome, Sesso, Eta, Altezza, Peso, Squadra, Anno, Sport, Evento, Medaglia)."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        headers = ["Nome", "Sesso", "Età", "Altezza", "Peso", "Squadra", "Anno", "Sport", "Evento", "Medaglia"]
                        rows = [format_row([
                            sol["Nome"], sol["Sesso"], sol["Eta"], sol["Altezza"], sol["Peso"],
                            sol["Squadra"], sol["Anno"], sol["Sport"], sol["Evento"], sol["Medaglia"]
                        ]) for sol in risultati]
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print(f"\nAtleta '{nome}' non trovato nella Knowledge Base.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {query}\nDettagli errore: {e}")

            elif scelta == "2":
                squadra = input("Inserisci nazionalità atleta (es. USA): ").strip().title()
                query = f"atleta(Nome, _, _, _, _, '{squadra}', _, _, _, _)."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        nomi_unici = {sol["Nome"] for sol in risultati}
                        print(f"\nAtleti della squadra '{squadra}':")
                        headers = ["Nome"]
                        rows = [[nome] for nome in sorted(nomi_unici)]
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print(f"\nNessun atleta trovato per la squadra '{squadra}'.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {query}\nDettagli errore: {e}")

            elif scelta == "3":
                sport = input("Inserisci il nome dello sport (es. Swimming): ").strip().title()
                query = f"atleta(Nome, _, _, _, _, _, _, '{sport}', Evento, Medaglia), Medaglia \\= 'none'."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        headers = ["Nome", "Evento", "Medaglia"]
                        rows = [[sol["Nome"], sol["Evento"], sol["Medaglia"]] for sol in risultati]
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print(f"\nNessuna medaglia trovata per lo sport '{sport}'.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {e}")

            elif scelta == "4":
                nome = input("Inserisci il nome dell'atleta: ").strip().title()
                query = f"atleta('{nome}', _, _, _, _, _, _, _, Evento, Medaglia), Medaglia \\= 'none'."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        headers = ["Evento", "Medaglia"]
                        rows = [[sol["Evento"], sol["Medaglia"]] for sol in risultati]
                        print(f"\nMigliori risultati per '{nome}':")
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print(f"\nNessun risultato trovato per '{nome}'.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {e}")

            elif scelta == "5":
                squadra = input("Inserisci la nazionalità (es. Italy): ").strip().title()
                query = f"findall(Sport, (atleta(_, _, _, _, _, '{squadra}', _, Sport, _, Medaglia), Medaglia \\= 'none'), Sports), sort(Sports, SortedSports)."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        headers = ["Sport"]
                        rows = [[sport] for sport in risultati[0]["SortedSports"]]
                        print(f"\nSport più popolari per '{squadra}':")
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print(f"\nNessuno sport trovato per '{squadra}'.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {e}")

            elif scelta == "6":
                tipo = input("Vuoi cercare l'atleta più giovane o anziano? (giovane/anziano): ").strip().lower()
                if tipo not in ["giovane", "anziano"]:
                    print("Scelta non valida. Inserisci 'giovane' o 'anziano'.")
                else:
                    query = f"atleta_piu_giovane_o_anziano({tipo}, Nome, Eta)."
                    try:
                        risultati = list(prolog.query(query))
                        if risultati:
                            nome = risultati[0]["Nome"]
                            eta = risultati[0]["Eta"]
                            
                            # Stampa il risultato in tabella
                            headers = ["Nome", "Età"]
                            rows = [[nome, eta]]
                            print(f"\nAtleta più {tipo}:")
                            print(tabulate(rows, headers=headers, tablefmt="grid"))
                        else:
                            print(f"\nNessun risultato trovato.")
                    except Exception as e:
                        print(f"Errore durante l'esecuzione della query: {query}\nDettagli errore: {e}")

            elif scelta == "7":
                squadra = input("Inserisci il nome della squadra (es. USA): ").strip().title()
                query = f"rapporto_medaglie_partecipanti('{squadra}', Ratio)."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        rapporto = risultati[0]["Ratio"]
                        print(f"\nRapporto medaglie/partecipanti per '{squadra}': {rapporto:.2f}")
                    else:
                        print(f"\nNessun risultato trovato per '{squadra}'.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {query}\nDettagli errore: {e}")

            elif scelta == "8":
                query = "findall((Anno, Conteggio), distribuzione_medaglie_per_edizione(Anno, Conteggio), Lista)."
                try:
                    risultati = list(prolog.query(query))
                    if risultati:
                        lista = risultati[0]["Lista"]  # Recupera i dati grezzi
                        print(f"Risultati grezzi da Prolog: {lista}")  # Debug per capire il formato
                        # Pulisci e converti ogni elemento della lista in una tuple
                        rows = [eval(item.strip(",").strip()) for item in lista]
                        headers = ["Anno", "Totale Medaglie"]
                        print("\nDistribuzione delle medaglie per edizione:")
                        print(tabulate(rows, headers=headers, tablefmt="grid"))
                    else:
                        print("\nNessuna distribuzione trovata.")
                except Exception as e:
                    print(f"Errore durante l'esecuzione della query: {query}\nDettagli errore: {e}")

            elif scelta == "9":
                print("Esco dall kb..")
                break

            else:
                print("Scelta non valida. Riprova.")

        except Exception as e:
            print(f"Errore inatteso: {e}")

if __name__ == "__main__":
    main()
