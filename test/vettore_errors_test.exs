defmodule VettoreErrorsTest do
  use ExUnit.Case, async: true
  import TestData
  alias Vettore.Embedding

  @moduletag :errors

  setup_all do
    {:ok, embeddings: load_embeddings()}
  end

  describe "invalid collection" do
    test "create_collection/4 with invalid distance", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert {:error, _} =
               Vettore.create_collection(db, "my_collection", length(e.vector), :unknown)
    end

    test "create_collection/4 with invalid keep_embeddings", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert {:error, _} =
               Vettore.create_collection(db, "my_collection", length(e.vector), :euclidean,
                 keep_embeddings: 1
               )
    end

    test "delete_collection/2 with invalid collection" do
      db = Vettore.new()
      assert {:error, _} = Vettore.delete_collection(db, "unknown")
    end

    test "insert/3 with invalid collection", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert {:error, _} =
               Vettore.insert(db, "unknown", %Embedding{
                 value: "my_id",
                 vector: e.vector,
                 metadata: %{"note" => "hello"}
               })
    end

    test "insert/3 with invalid metadata", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert {:error,
              "invalid metadata entry \"note\" => [1, 2, 4], metadata must be string => string"} =
               Vettore.insert(db, "unknown", %Embedding{
                 value: "my_id",
                 vector: e.vector,
                 metadata: %{"note" => [1, 2, 4]}
               })
    end

    test "insert!/3 with invalid metadata" do
      db = Vettore.new()

      assert_raise ArgumentError,
                   "invalid metadata entry \"note\" => [1, 2, 4], metadata must be string => string",
                   fn ->
                     Vettore.insert!(db, "unknown", %Embedding{
                       value: "my_id",
                       vector: [1.0, 2.0, 3.0],
                       metadata: %{"note" => [1, 2, 4]}
                     })
                   end
    end

    test "batch/3 with invalid collection", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert {:error, _} =
               Vettore.batch(db, "unknown", [
                 %Embedding{
                   value: "my_id",
                   vector: e.vector,
                   metadata: %{"note" => "hello"}
                 }
               ])
    end

    test "batch!/3 with invalid collection", %{embeddings: [e | _]} do
      db = Vettore.new()

      assert_raise ArgumentError,
                   "each item must be %Vettore.Embedding{} with valid fields, got: %Vettore.Embedding{value: \"my_id\", vector: \"Laravel,PHP,Bug Fixing,Web Development\", metadata: %{\"note\" => \"ok\"}}",
                   fn ->
                     Vettore.batch!(db, "unknown", [
                       %Embedding{
                         value: "my_id",
                         vector: e.value,
                         metadata: %{"note" => "ok"}
                       }
                     ])
                   end
    end

    test "get_by_value/3 with invalid collection" do
      db = Vettore.new()
      assert {:error, _} = Vettore.get_by_value(db, "unknown", "my_id")
    end

    test "get_by_vector/3 with invalid collection", %{embeddings: [e | _]} do
      db = Vettore.new()
      assert {:error, _} = Vettore.get_by_vector(db, "unknown", e.vector)
    end

    test "delete/3 with invalid collection" do
      db = Vettore.new()
      assert {:error, _} = Vettore.delete(db, "unknown", "my_id")
    end

    test "get_all/2 with invalid collection" do
      db = Vettore.new()
      assert {:error, _} = Vettore.get_all(db, "unknown")
    end

    test "similarity_search/4 with invalid collection", %{embeddings: [e | _]} do
      db = Vettore.new()
      assert {:error, _} = Vettore.similarity_search(db, "unknown", e.vector, limit: 1)
    end

    test "rerank/4 with invalid collection" do
      db = Vettore.new()

      assert {:error, _} =
               Vettore.rerank(db, "unknown", [{"my_id", 0.0}, {"my_id2", 0.0}, {"my_id3", 0.0}],
                 limit: 1
               )
    end
  end
end
