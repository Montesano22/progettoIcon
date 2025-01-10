# OlympicX

## Introduzione
Questo progetto analizza e gestisce un dataset delle Olimpiadi utilizzando tecniche di apprendimento supervisionato e non supervisionato, oltre ad esplorare ontologie e KnoledgeBase.

Il progetto è progettato per essere semplice da configurare e avviare, utilizzando Python e le sue librerie scientifiche come `pandas`, `seaborn`, e `scikit-learn`. 

E' necessario avviare il file in base alle tecniche di machine learning o argomenti che ci interessano.

---

## Requisiti
- Python 3.8 o superiore
- Ambiente virtuale Python (opzionale ma consigliato)

---

## Installazione delle dipendenze
1. Clona o scarica il progetto nella tua directory preferita.
2. Apri un terminale e spostati nella directory del progetto:
   ```bash
   cd path/del/progetto
   ```
3. (Opzionale) Crea ed attiva un ambiente virtuale:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Su Windows
   source venv/bin/activate  # Su Linux/Mac
   ```
4. Installa le dipendenze richieste:
   ```bash
   pip install -r requirements.txt
   ```

---

## Come avviare il progetto
### Metodo 1: Avvio tramite terminale
1. Assicurati di trovarti nella directory principale del progetto.
2. Esegui il comando:
   ```bash
   python modelloDiInteresse.py
   ```
3. Segui le istruzioni visualizzate nel terminale per interagire con il menu principale del progetto.



---

## Struttura del Progetto
- **`apprendimentoSupervisionato.py`**: Contiene il codice per l'apprendimento supervisionato, inclusi random forest, support vector machine e gradient boosting.
- **`apprendimentoNonSupervisionato.py`**: Contiene il codice per l'apprendimento non supervisionato, inclusi clustering con KMeans e rilevamento di anomalie.
- **`mainOnthology`**: Consente di esplorare l'ontologia.
- **`knowledgeBase.py`**: Consente di esplorare il dominio effettuando query.
- **`requirements.txt`**: Elenco delle dipendenze Python necessarie per il progetto.

---

## Note Finali
- Assicurati di aver configurato correttamente il tuo ambiente Python.
- Abbiamo testato il codice su diversi sistemi operativi (potrebberro volerci librerie aggiuntive su macos/linux)
- Se riscontri problemi durante l'avvio, verifica che tutte le librerie siano installate correttamente con:
  ```bash
  pip install -r requirements.txt
  ```

Ora sei pronto per esplorare il progetto OlympicX e analizzare i dati in modo interattivo!
