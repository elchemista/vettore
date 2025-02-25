## NIF Functions

All core functions are accessible in Elixir via `Vettore.*` calls. Their **return values** (on success) now include more information:

1. **`create_collection(db, name, dimension, distance)`**  
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Creates a new collection in the database with a specified dimension and distance metric.

2. **`delete_collection(db, name)`**  
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Deletes an existing collection (by name).

3. **`insert_embedding(db, collection_name, embedding_struct)`**  
   Returns `{:ok, embedding_id}` or `{:error, reason}`.

   - Inserts a single embedding (with an ID, vector, and optional metadata).

4. **`insert_embeddings(db, collection_name, [embedding_structs])`**  
   Returns `{:ok, list_of_inserted_ids}` or `{:error, reason}`.

   - **Batch insertion**: Insert a list of embeddings in one call.
   - If any embedding fails (dimension mismatch, duplicate ID, etc.), an error is returned immediately and the rest are not inserted.

5. **`get_embeddings(db, collection_name)`**  
   Returns `{:ok, list_of({id, vector, metadata})}` or `{:error, reason}`.

   - Retrieves all embeddings from the specified collection.

6. **`get_embedding_by_id(db, collection_name, id)`**  
   Returns `{:ok, %Vettore.Embedding{}}` or `{:error, reason}`.

   - Looks up a single embedding by its ID.

7. **`similarity_search(db, collection_name, query_vector, k)`**  
   Returns `{:ok, list_of({id, score})}` or `{:error, reason}`.

   - Performs a similarity or distance search with the given query vector, returning the top‑k results.

8. **`new_db()`**  
   Returns a **DB resource** (reference to the underlying Rust `CacheDB`).

---