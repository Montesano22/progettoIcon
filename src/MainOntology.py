from owlready2 import *

def main_ontology():
    print("\nBENVENUTO NELL'ONTOLOGIA OLIMPICA")

    # Caricamento dell'ontologia
    try:
        ontology_path = 'ontologiaOlimpiadi.owl'  # Sostituire con il nome corretto del file
        olympics_ontology = get_ontology(ontology_path).load()
        print("\[INFO] Ontologia caricata correttamente!")
    except Exception as e:
        print(f"[ERRORE] errore nel caricamento dell'ontologia: {e}")
        return

    while True:
        print("\nSeleziona un'operazione:")
        print("1) Visualizzazione Classi")
        print("2) Visualizzazione proprietà d'oggetto")
        print("3) Visualizzazione proprietà dei dati")
        print("4) Esegui query")
        print("5) Esci dall'Ontologia")

        menu_answer = input("\nInserisci un valore (1-5): ")

        if menu_answer == '1':
            print("\nCLASSI PRESENTI NELL'ONTOLOGIA:")
            for cls in olympics_ontology.classes():
                print(f"- {cls.name}")

        elif menu_answer == '2':
            print("\nPROPRIETÁ D'OGGETTO PRESENTI NELL'ONTOLOGIA:")
            for prop in olympics_ontology.object_properties():
                label = getattr(prop, "label", None)  # Controlla se esiste un'etichetta leggibile
                if label:
                    print(f"- {label[0]}")
                else:
                    print(f"- {prop.name}")  # Usa il nome interno come fallback

        elif menu_answer == '3':
            print("\nPROPRIETÁ DEI DATI PRESENTI NELL'ONTOLOGIA:")
            for prop in olympics_ontology.data_properties():
                label = getattr(prop, "label", None)  # Controlla se esiste un'etichetta leggibile
                if label:
                    print(f"- {label[0]}")
                else:
                    print(f"- {prop.name}")  # Usa il nome interno come fallback

        elif menu_answer == '4':
            while True:
                print("\nSELEZIONA UNA QUERY:")
                print("1) Lista degli atleti presenti nell'ontologia")
                print("2) Lista degli sport presenti nell'ontologia")
                print("3) Lista delle squadre partecipanti")
                print("4) Torna indietro")

                query_choice = input("\nInserisci un valore (1-4): ")

                if query_choice == '1':
                    print("\nELENCO DI TUTTI GLI ATLETI:")
                    athlete_class = olympics_ontology.search_one(iri="*Athlete")
                    if athlete_class:
                        athletes = olympics_ontology.search(is_a=athlete_class)
                        if not athletes:
                            print("Nessun individuo trovato per la classe 'Athlete'.")
                        for athlete in athletes:
                            label = getattr(athlete, "label", None)
                            name = getattr(athlete, "hasName", None)

                            if label:
                                print(f"- {label[0]}")
                            elif name:
                                print(f"- {name[0]}")
                            else:
                                print(f"- {athlete.name}")
                    else:
                        print("Errore: la classe 'Athlete' non è stata trovata nell'ontologia.")

                elif query_choice == '2':
                    print("\nSPORT PRESENTI NELL'ONTOLOGIA:")
                    # Cerca specificamente la classe Sport
                    sport_class = next((cls for cls in olympics_ontology.classes() if "Sport" in cls.name), None)
                    if sport_class:
                        # Usa search() per cercare individui della classe
                        sports = olympics_ontology.search(is_a=sport_class)
                        if not sports:
                            print("Nessun individuo trovato per la classe 'Sport'.")
                        else:
                            for sport in sports:
                                label = getattr(sport, "label", None)
                                if label:
                                    print(f"- {label[0]}")
                                else:
                                    print(f"- {sport.name}")
                    else:
                        print("Errore: la classe 'Sport' non è stata trovata nell'ontologia.")

                elif query_choice == '3':
                    print("\nSQUADRE PARTECIPANTI:")
                    team_class = olympics_ontology.search_one(iri="*Team")
                    if team_class:
                        teams = olympics_ontology.search(is_a=team_class)
                        if not teams:
                            print("Nessun individuo trovato per la classe 'Team'.")
                        for team in teams:
                            label = getattr(team, "label", None)
                            name = getattr(team, "hasName", None)

                            if label:
                                print(f"- {label[0]}")
                            elif name:
                                if isinstance(name[0], str) and name[0].startswith("xsd:string"):
                                    print(f"- {name[0].replace('xsd:string ', '')}")
                                else:
                                    print(f"- {name[0]}")
                            else:
                                print(f"- {team.name}")
                    else:
                        print("Errore: la classe 'Team' non è stata trovata nell'ontologia.")

                elif query_choice == '4':
                    break

                else:
                    print("Valore non valido. Riprova.")

        elif menu_answer == '5':
            print("\nUscita dall'ontologia. Arrivederci!")
            break

        else:
            print("Valore non valido. Riprova.")

if __name__ == "__main__":
    main_ontology()
