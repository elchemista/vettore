alias Vettore.{Embedding, Store.ETS}

count = String.to_integer(System.get_env("VETTORE_BENCH_BATCH", "10000"))
dimensions = String.to_integer(System.get_env("VETTORE_BENCH_DIMENSIONS", "16"))
vector = List.duplicate(1.0, dimensions)

embeddings =
  for index <- 1..count do
    %Embedding{id: Integer.to_string(index), vector: vector}
  end

records = Enum.map(embeddings, &{{:record, &1.id}, &1})

timed = fn label, fun ->
  {microseconds, _result} = :timer.tc(fun)
  seconds = microseconds / 1_000_000
  throughput = if seconds == 0.0, do: :infinity, else: round(count / seconds)
  IO.puts("#{label}: #{Float.round(seconds, 4)}s (#{throughput} records/s)")
end

direct_single =
  :ets.new(:vettore_direct_write_benchmark, [
    :set,
    :public,
    write_concurrency: true
  ])

timed.("direct public ETS, one insert per record", fn ->
  Enum.each(records, fn record -> true = :ets.insert(direct_single, record) end)
end)

direct_batch =
  :ets.new(:vettore_direct_batch_benchmark, [
    :set,
    :public,
    write_concurrency: true
  ])

timed.("direct public ETS, one batched insert", fn ->
  true = :ets.insert_new(direct_batch, records)
end)

{:ok, raw_owner_single} = ETS.new(%{})

timed.("supervised owner, one call per record", fn ->
  Enum.each(records, fn record ->
    true = Vettore.ETSOwner.insert_new(raw_owner_single.owner, record)
  end)
end)

{:ok, raw_owner_batch} = ETS.new(%{})

timed.("supervised owner, one raw batched call", fn ->
  true = Vettore.ETSOwner.insert_new(raw_owner_batch.owner, records)
end)

{:ok, validated_batch} = ETS.new(%{})

timed.("Store.ETS.put_many with validation", fn ->
  :ok = ETS.put_many(validated_batch, embeddings)
end)

true = :ets.delete(direct_single)
true = :ets.delete(direct_batch)
:ok = ETS.close(raw_owner_single)
:ok = ETS.close(raw_owner_batch)
:ok = ETS.close(validated_batch)
