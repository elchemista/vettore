## NIF Functions

All core functions are accessible in Elixir via `Vettore.*` calls. Their **return values** (on success) now include more information:

1. **`create_collection(db, name, dimension, distance, keep_embeddings)`**
   * `:keep_embeddings` has boolean value valid only on "binary" collection
     
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Creates a new collection in the database with a specified dimension and distance metric.

3. **`delete_collection(db, name)`**  
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Deletes an existing collection (by name).

4. **`insert_embedding(db, collection_name, embedding_struct)`**  
   Returns `{:ok, embedding_id}` or `{:error, reason}`.

   - Inserts a single embedding (with an ID, vector, and optional metadata).

5. **`insert_embeddings(db, collection_name, [embedding_structs])`**  
   Returns `{:ok, list_of_inserted_ids}` or `{:error, reason}`.

   - **Batch insertion**: Insert a list of embeddings in one call.
   - If any embedding fails (dimension mismatch, duplicate ID, etc.), an error is returned immediately and the rest are not inserted.

6. **`get_embeddings(db, collection_name)`**  
   Returns `{:ok, list_of({id, vector, metadata})}` or `{:error, reason}`.

   - Retrieves all embeddings from the specified collection.

7. **`get_embedding_by_id(db, collection_name, id)`**  
   Returns `{:ok, %Vettore.Embedding{}}` or `{:error, reason}`.

   - Looks up a single embedding by its ID.

8. **`similarity_search(db, collection_name, query_vector, k, filters)`**  
   * filters has as parameters `:limit`, `:filter`. ex: `[limit: 2, filter: %{"category" => "test"}]`
   Returns `{:ok, list_of({id, score})}` or `{:error, reason}`.

   - Performs a similarity or distance search with the given query vector, returning the top‑k results.

9. **`new_db()`**  
   Returns a **DB resource** (reference to the underlying Rust `CacheDB`).

10. **`mmr_rerank(db, collection_name, initial_results, opts)`**  
    Returns `{:ok, [{id, mmr_score}, ...]}` or `{:error, reason}`.
    
    - Re-ranks a list of `{id, score}` pairs (e.g., from a previous `similarity_search/4` call) using **Maximal Marginal Relevance (MMR)**.  
    This helps select up to `:limit` items that are both highly relevant (based on their initial score) and also sufficiently distinct from each other.
    
    Optional parameters:
    - `limit: k` — How many items to keep in the final MMR‑ranked list (default 10).
    - `alpha: float` — MMR balancing factor in `[0..1]`. Closer to `1.0` places more emphasis on each candidate’s original score; closer to `0.0` emphasizes diversity.

---
