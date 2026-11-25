# Piano di lavoro per Vettore 0.3.7

## Correzioni

- [ ] **Validare `limit` nelle ricerche adattive.** In `funnel_search/3` e
  `quantized_search/3`, calcolare il numero di candidati dopo aver validato il limite.
  Con `limit: "10"` entrambe devono restituire `{:error, :invalid_limit}`, anche quando
  il chiamante specifica `candidates`. [Collection](lib/vettore/collection.ex)

- [ ] **Liberare la matrice Flat alla chiusura.** Rilasciare la memoria riservata a
  vettori, ID e posizioni quando si chiama `close/1`, anche se l'ultimo record è già
  stato cancellato. Un processo che conserva il riferimento alla collezione chiusa
  deve poter rilasciare subito i buffer tramite `close/1`. [Indice Flat](native/vettore/src/flat.rs)

- [ ] **Proteggere L2 e L2Squared su GPU.** Tra `[1.0e25, 0.0]` e `[1.0e25, 1.0]`
  entrambe le distanze valgono `1.0`. Dopo il ridimensionamento, il quadrato della
  differenza può arrotondarsi a zero in f32. Correggere il calcolo o rilevare il caso
  e applicare il fallback configurato nelle metriche dirette, nel top-k Flat e nel
  reranking GPU. Un risultato a distanza zero deve precedere quello a distanza uno.
  [Calcolo GPU](native/vettore/src/gpu.rs) · [Preparazione dei vettori](native/vettore/src/gpu_math.rs)

- [ ] **Associare il punteggio alla versione del record.** Assegnare una nuova revisione
  anche quando un ID viene cancellato e reinserito. Farla viaggiare con ID e punteggio
  da Flat e HNSW fino alla lettura da ETS. Se la revisione è cambiata, scartare il
  risultato obsoleto o ripetere la ricerca con un limite ai tentativi. Il punteggio,
  il valore e i metadati restituiti devono riferirsi alla stessa revisione. Per gli
  snapshot esistenti, generare le revisioni quando si ricostruisce l'indice.
  [Lettura dei risultati](lib/vettore/index.ex) · [Store ETS](lib/vettore/store/ets.ex)

## Prestazioni

- [ ] **Ridurre le copie nella ricerca quantizzata.** Per la selezione tramite bit di
  segno, leggere da ETS solo ID, revisione e parole binarie. Recuperare i vettori
  completi dei candidati per il reranking. Gestire gli snapshot privi dei bit
  precalcolati e gli store personalizzati. Conservare la semantica di Hamming e
  Jaccard sui valori diversi da zero. A parità di candidati, confrontare risultati,
  memoria allocata per query e latenza con il percorso attuale.
  [Selezione dei candidati](lib/vettore/collection.ex)

## Nuove API

- [ ] **Aggiungere `upsert/2` atomico.** Inserire gli ID nuovi e aggiornare quelli
  esistenti. Validare l'embedding prima della sostituzione e usare la revisione per
  coordinare lettori e scrittori. Un vettore di dimensioni errate o un errore durante
  l'aggiornamento dell'indice devono lasciare il record precedente leggibile e
  ricercabile. Conservare `{:error, :duplicate_id}` per chi usa `put/2`.
  [API pubblica](lib/vettore.ex) · [Scritture indicizzate](lib/vettore/store/ets.ex)

- [ ] **Filtrare per metadati durante la ricerca.** Introdurre `filter: %{lingua: "it"}`
  in `search/3`, con corrispondenza esatta su tutte le chiavi richieste. Flat deve
  calcolare il top-k tra i record idonei. HNSW può attraversare nodi esclusi dal
  filtro e raccogliere solo nodi idonei per i risultati. Se i primi dieci record
  globali sono in inglese, Flat deve comunque restituire i dieci italiani più vicini,
  quando presenti. Consultare il prototipo del branch locale `feature/metadata-filter`.
  [Ricerca Flat](lib/vettore/index/flat.ex) · [Ricerca HNSW](lib/vettore/index/hnsw.ex)
