alias Vettore.{Embedding, Store.ETS}

read_count = String.to_integer(System.get_env("VETTORE_BENCH_READS", "1000000"))

workers =
  System.get_env("VETTORE_BENCH_READ_WORKERS", Integer.to_string(System.schedulers_online()))
  |> String.to_integer()

if read_count <= 0 or workers <= 0 do
  raise ArgumentError, "VETTORE_BENCH_READS and VETTORE_BENCH_READ_WORKERS must be positive"
end

{:ok, state} = ETS.new(%{})
:ok = ETS.put(state, %Embedding{id: "key", vector: [1.0]})

timed = fn label, operations, fun ->
  {microseconds, _result} = :timer.tc(fun)
  seconds = microseconds / 1_000_000
  throughput = if seconds == 0.0, do: :infinity, else: round(operations / seconds)
  IO.puts("#{label}: #{Float.round(seconds, 4)}s (#{throughput} reads/s)")
end

timed.("direct protected ETS, sequential", read_count, fn ->
  Enum.each(1..read_count, fn _ ->
    [{{:record, "key"}, %Embedding{}}] = :ets.lookup(state.table, {:record, "key"})
  end)
end)

timed.("Store.ETS.get, sequential", read_count, fn ->
  Enum.each(1..read_count, fn _ ->
    {:ok, %Embedding{}} = ETS.get(state, "key")
  end)
end)

reads_per_worker = div(read_count + workers - 1, workers)
parallel_reads = reads_per_worker * workers

timed.("Store.ETS.get, #{workers} readers", parallel_reads, fn ->
  1..workers
  |> Task.async_stream(
    fn _worker ->
      Enum.each(1..reads_per_worker, fn _ ->
        {:ok, %Embedding{}} = ETS.get(state, "key")
      end)
    end,
    max_concurrency: workers,
    ordered: false,
    timeout: :infinity
  )
  |> Stream.run()
end)

:ok = ETS.close(state)
