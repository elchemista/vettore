# These tests are deliberately excluded from normal runs. Examples:
#
#   VETTORE_RUN_ENDURANCE=1 \
#   VETTORE_SOAK_SECONDS=14400 \
#   VETTORE_SOAK_DATASET_SIZE=3000 \
#   VETTORE_BUILD=1 \
#   mix test test/vector_endurance_test.exs --only soak --trace
#
#   VETTORE_RUN_ENDURANCE=1 \
#   VETTORE_BENCH_SECONDS=14400 \
#   VETTORE_BENCH_DATASET_SIZE=3000 \
#   VETTORE_BUILD=1 \
#   mix test test/vector_endurance_test.exs --only endurance_benchmark --trace
#
# Add VETTORE_ENDURANCE_LIVE_FASTEMBED=1 to regenerate the vectors with
# ex_fastembed before either run. By default the committed artifact generated
# by ex_fastembed is used, keeping long runs deterministic and offline.

Code.require_file("support/fastembed_fixture.ex", __DIR__)

defmodule VettoreEnduranceTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Embedding, Result}
  alias Vettore.Test.FastembedFixture

  @moduletag skip: System.get_env("VETTORE_RUN_ENDURANCE") != "1"
  @moduletag timeout: :infinity

  @hnsw_options [
    m: 16,
    m0: 32,
    ef_construction: 200,
    ef_search: 200,
    max_level: 16
  ]

  @benchmark_scenarios [:flat, :hnsw, :funnel, :quantized, :hybrid]

  @latency_buckets_us [
    10,
    25,
    50,
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    5_000_000
  ]

  @f32_max 3.402_823_466_385_288_6e38

  @tag :soak
  test "concurrent searches, churn, and snapshots remain consistent for hours" do
    config = soak_config!()
    fixture = fastembed_fixture!()
    dataset = build_dataset!(fixture, config.dataset_size, config.seed)
    unique = System.unique_integer([:positive, :monotonic])

    flat = collection!("endurance-flat-#{unique}", dataset.dimensions, :flat)
    on_exit(fn -> Collection.close(flat) end)

    hnsw = collection!("endurance-hnsw-#{unique}", dataset.dimensions, :hnsw)
    on_exit(fn -> Collection.close(hnsw) end)

    put_many!(flat, dataset.embeddings)
    put_many!(hnsw, dataset.embeddings)

    started_ms = monotonic_ms()

    context = %{
      flat: flat,
      hnsw: hnsw,
      id_tuple: dataset.embeddings |> Enum.map(& &1.id) |> List.to_tuple(),
      expected_ids: MapSet.new(dataset.embeddings, & &1.id),
      query_tuple: List.to_tuple(dataset.queries),
      limit: config.limit,
      config: config
    }

    stats = %{
      started_ms: started_ms,
      last_report_ms: started_ms,
      rounds: 0,
      search_pairs: 0,
      mutations: 0,
      snapshots: 0,
      flat_search_us: 0,
      hnsw_search_us: 0,
      minimum_audit_recall: 1.0
    }

    deadline_ms = started_ms + config.duration_seconds * 1_000
    final_stats = run_soak(context, deadline_ms, stats, 1)
    final_recall = audit_collections!(context)
    final_stats = merge_audit_recall(final_stats, final_recall)

    print_soak_report(final_stats, true)

    assert final_stats.rounds > 0
    assert final_stats.search_pairs > 0
    assert final_stats.mutations > 0
  end

  @tag :endurance_benchmark
  test "real-embedding search modes retain recall and stable latency for hours" do
    config = benchmark_config!()
    fixture = fastembed_fixture!()
    dataset = build_dataset!(fixture, config.dataset_size, config.seed)
    unique = System.unique_integer([:positive, :monotonic])

    flat = collection!("benchmark-flat-#{unique}", dataset.dimensions, :flat)
    on_exit(fn -> Collection.close(flat) end)

    hnsw = collection!("benchmark-hnsw-#{unique}", dataset.dimensions, :hnsw)
    on_exit(fn -> Collection.close(hnsw) end)

    put_many!(flat, dataset.embeddings)
    put_many!(hnsw, dataset.embeddings)

    started_ms = monotonic_ms()

    context = %{
      flat: flat,
      hnsw: hnsw,
      query_tuple: List.to_tuple(dataset.queries),
      dimensions: dataset.dimensions,
      limit: config.limit,
      candidates: config.candidates,
      config: config
    }

    stats = %{
      started_ms: started_ms,
      last_report_ms: started_ms,
      rounds: 0,
      scenarios: empty_scenario_stats(),
      recall: %{}
    }

    deadline_ms = started_ms + config.duration_seconds * 1_000
    final_stats = run_benchmark(context, deadline_ms, stats, 1)
    final_recall = audit_search_modes!(context)
    final_stats = %{final_stats | recall: final_recall}

    print_benchmark_report(final_stats, true)

    assert final_stats.rounds > 0
    assert total_benchmark_operations(final_stats.scenarios) > 0
  end

  @typep soak_config :: %{
           duration_seconds: pos_integer(),
           dataset_size: pos_integer(),
           readers: pos_integer(),
           operations_per_round: pos_integer(),
           mutations_per_round: pos_integer(),
           audit_every_rounds: pos_integer(),
           snapshot_every_rounds: non_neg_integer(),
           report_every_ms: pos_integer(),
           audit_queries: pos_integer(),
           limit: pos_integer(),
           minimum_recall: float(),
           maximum_memory_bytes: non_neg_integer(),
           seed: non_neg_integer()
         }

  @typep benchmark_config :: %{
           duration_seconds: pos_integer(),
           dataset_size: pos_integer(),
           workers: pos_integer(),
           operations_per_round: pos_integer(),
           audit_every_rounds: pos_integer(),
           report_every_ms: pos_integer(),
           audit_queries: pos_integer(),
           limit: pos_integer(),
           candidates: pos_integer(),
           minimum_hnsw_recall: float(),
           maximum_memory_bytes: non_neg_integer(),
           seed: non_neg_integer()
         }

  @typep dataset :: %{
           dimensions: pos_integer(),
           embeddings: [Embedding.t()],
           queries: [[float()]]
         }

  @typep latency_stats :: %{
           count: non_neg_integer(),
           total_us: non_neg_integer(),
           maximum_us: non_neg_integer(),
           buckets: %{optional(pos_integer() | :infinity) => non_neg_integer()}
         }

  @spec soak_config!() :: soak_config()
  defp soak_config! do
    dataset_size = positive_integer_env!("VETTORE_SOAK_DATASET_SIZE", 3_000)
    limit = positive_integer_env!("VETTORE_SOAK_LIMIT", 10)

    ensure_limit_fits!(limit, dataset_size, "VETTORE_SOAK_LIMIT")

    %{
      duration_seconds: positive_integer_env!("VETTORE_SOAK_SECONDS", 3_600),
      dataset_size: dataset_size,
      readers: positive_integer_env!("VETTORE_SOAK_READERS", max(System.schedulers_online(), 1)),
      operations_per_round: positive_integer_env!("VETTORE_SOAK_OPERATIONS", 50),
      mutations_per_round: positive_integer_env!("VETTORE_SOAK_MUTATIONS", 25),
      audit_every_rounds: positive_integer_env!("VETTORE_SOAK_AUDIT_EVERY", 5),
      snapshot_every_rounds: non_negative_integer_env!("VETTORE_SOAK_SNAPSHOT_EVERY", 10),
      report_every_ms: positive_integer_env!("VETTORE_SOAK_REPORT_SECONDS", 60) * 1_000,
      audit_queries: positive_integer_env!("VETTORE_SOAK_AUDIT_QUERIES", 20),
      limit: limit,
      minimum_recall: probability_env!("VETTORE_SOAK_MIN_RECALL", 0.8),
      maximum_memory_bytes:
        non_negative_integer_env!("VETTORE_SOAK_MAX_MEMORY_MB", 0) * 1_024 * 1_024,
      seed: non_negative_integer_env!("VETTORE_SOAK_SEED", 20_260_812)
    }
  end

  @spec benchmark_config!() :: benchmark_config()
  defp benchmark_config! do
    dataset_size = positive_integer_env!("VETTORE_BENCH_DATASET_SIZE", 3_000)
    limit = positive_integer_env!("VETTORE_BENCH_LIMIT", 10)
    candidates = positive_integer_env!("VETTORE_BENCH_CANDIDATES", min(dataset_size, 500))

    ensure_limit_fits!(limit, dataset_size, "VETTORE_BENCH_LIMIT")
    ensure_limit_fits!(limit, candidates, "VETTORE_BENCH_LIMIT")

    %{
      duration_seconds: positive_integer_env!("VETTORE_BENCH_SECONDS", 3_600),
      dataset_size: dataset_size,
      workers: positive_integer_env!("VETTORE_BENCH_WORKERS", max(System.schedulers_online(), 1)),
      operations_per_round: positive_integer_env!("VETTORE_BENCH_OPERATIONS", 100),
      audit_every_rounds: positive_integer_env!("VETTORE_BENCH_AUDIT_EVERY", 10),
      report_every_ms: positive_integer_env!("VETTORE_BENCH_REPORT_SECONDS", 60) * 1_000,
      audit_queries: positive_integer_env!("VETTORE_BENCH_AUDIT_QUERIES", 20),
      limit: limit,
      candidates: candidates,
      minimum_hnsw_recall: probability_env!("VETTORE_BENCH_MIN_HNSW_RECALL", 0.8),
      maximum_memory_bytes:
        non_negative_integer_env!("VETTORE_BENCH_MAX_MEMORY_MB", 0) * 1_024 * 1_024,
      seed: non_negative_integer_env!("VETTORE_BENCH_SEED", 20_260_812)
    }
  end

  @spec fastembed_fixture!() :: map()
  defp fastembed_fixture! do
    if System.get_env("VETTORE_ENDURANCE_LIVE_FASTEMBED") == "1" do
      IO.puts("Generating fresh real embeddings with ex_fastembed...")
      FastembedFixture.generate!()
    else
      FastembedFixture.load!()
    end
  end

  @spec build_dataset!(map(), pos_integer(), non_neg_integer()) :: dataset()
  defp build_dataset!(fixture, dataset_size, seed) do
    embeddings = expand_embeddings(fixture.documents, dataset_size, seed)

    fixture_queries = Enum.map(fixture.queries, & &1.vector)
    document_queries = sampled_document_vectors(embeddings, max(32 - length(fixture_queries), 1))

    %{
      dimensions: fixture.dimensions,
      embeddings: embeddings,
      queries: fixture_queries ++ document_queries
    }
  end

  @spec expand_embeddings([map()], pos_integer(), non_neg_integer()) :: [Embedding.t()]
  defp expand_embeddings(documents, dataset_size, seed) do
    document_tuple = List.to_tuple(documents)
    document_count = tuple_size(document_tuple)

    Enum.map(0..(dataset_size - 1), fn index ->
      source = elem(document_tuple, rem(index, document_count))
      generation = div(index, document_count)

      %Embedding{
        id: "endurance-#{index}",
        value: source.text,
        vector: perturb_and_normalize(source.vector, index, seed),
        metadata: %{
          source_id: source.id,
          category: source.category,
          generation: generation
        }
      }
    end)
  end

  @spec perturb_and_normalize([float()], non_neg_integer(), non_neg_integer()) :: [float()]
  defp perturb_and_normalize(vector, record_index, seed) do
    perturbed =
      vector
      |> Enum.with_index()
      |> Enum.map(fn {value, dimension_index} ->
        phase = (seed + record_index + 1) * (dimension_index + 1)
        value + :math.sin(phase) * 1.0e-4
      end)

    norm =
      perturbed
      |> Enum.reduce(0.0, fn value, sum -> sum + value * value end)
      |> :math.sqrt()

    Enum.map(perturbed, &(&1 / norm))
  end

  @spec sampled_document_vectors([Embedding.t()], pos_integer()) :: [[float()]]
  defp sampled_document_vectors(embeddings, count) do
    stride = max(div(length(embeddings), count), 1)

    embeddings
    |> Enum.take_every(stride)
    |> Enum.take(count)
    |> Enum.map(& &1.vector)
  end

  @spec collection!(String.t(), pos_integer(), :flat | :hnsw) :: Collection.t()
  defp collection!(name, dimensions, :flat) do
    case Collection.new(
           name: name,
           dimensions: dimensions,
           metric: :cosine,
           normalize: :l2,
           score: :raw,
           index: :flat
         ) do
      {:ok, collection} -> collection
      {:error, reason} -> raise "could not create flat collection: #{inspect(reason)}"
    end
  end

  defp collection!(name, dimensions, :hnsw) do
    case Collection.new(
           name: name,
           dimensions: dimensions,
           metric: :cosine,
           normalize: :l2,
           score: :raw,
           index: :hnsw,
           index_options: @hnsw_options
         ) do
      {:ok, collection} -> collection
      {:error, reason} -> raise "could not create HNSW collection: #{inspect(reason)}"
    end
  end

  @spec put_many!(Collection.t(), [Embedding.t()]) :: :ok
  defp put_many!(collection, embeddings) do
    case Collection.put_many(collection, embeddings) do
      :ok -> :ok
      {:error, reason} -> raise "could not populate endurance collection: #{inspect(reason)}"
    end
  end

  @spec run_soak(map(), integer(), map(), pos_integer()) :: map()
  defp run_soak(context, deadline_ms, stats, round) do
    if monotonic_ms() >= deadline_ms do
      stats
    else
      {round_stats, audit_recall} = run_soak_round(context, round)

      stats =
        stats
        |> merge_soak_round(round_stats)
        |> merge_audit_recall(audit_recall)
        |> maybe_print_soak_report(context.config.report_every_ms)

      run_soak(context, deadline_ms, stats, round + 1)
    end
  end

  @spec run_soak_round(map(), pos_integer()) :: {map(), float() | nil}
  defp run_soak_round(context, round) do
    reader_context = %{
      flat: context.flat,
      hnsw: context.hnsw,
      query_tuple: context.query_tuple,
      limit: context.limit,
      operations: context.config.operations_per_round,
      round: round
    }

    writer_context = %{
      flat: context.flat,
      hnsw: context.hnsw,
      id_tuple: context.id_tuple,
      operations: context.config.mutations_per_round,
      round: round
    }

    reader_tasks =
      Enum.map(1..context.config.readers, fn worker ->
        Task.async(fn -> run_soak_reader(reader_context, worker) end)
      end)

    writer_task = Task.async(fn -> run_soak_writer(writer_context) end)
    snapshot_task = maybe_start_snapshot_task(context, round)

    reader_reports = Task.await_many(reader_tasks, :infinity)
    mutations = Task.await(writer_task, :infinity)
    snapshots = await_optional_task(snapshot_task)

    audit_recall =
      if rem(round, context.config.audit_every_rounds) == 0 do
        audit_collections!(context)
      end

    round_stats =
      Enum.reduce(reader_reports, empty_soak_round(), fn report, acc ->
        %{
          acc
          | search_pairs: acc.search_pairs + report.search_pairs,
            flat_search_us: acc.flat_search_us + report.flat_search_us,
            hnsw_search_us: acc.hnsw_search_us + report.hnsw_search_us
        }
      end)

    {%{round_stats | mutations: mutations, snapshots: snapshots}, audit_recall}
  end

  @spec run_soak_reader(map(), pos_integer()) :: map()
  defp run_soak_reader(context, worker) do
    query_count = tuple_size(context.query_tuple)

    Enum.reduce(0..(context.operations - 1), empty_reader_report(), fn operation, report ->
      query_index = rem(context.round * 1_009 + worker * 101 + operation, query_count)
      query = elem(context.query_tuple, query_index)

      {flat_us, exact_results} = timed_search!(context.flat, query, context.limit)
      {hnsw_us, hnsw_results} = timed_search!(context.hnsw, query, context.limit)

      validate_results!(exact_results, context.limit)
      validate_results!(hnsw_results, context.limit)

      %{
        search_pairs: report.search_pairs + 1,
        flat_search_us: report.flat_search_us + flat_us,
        hnsw_search_us: report.hnsw_search_us + hnsw_us
      }
    end)
  end

  @spec run_soak_writer(map()) :: pos_integer()
  defp run_soak_writer(context) do
    embedding_count = tuple_size(context.id_tuple)

    Enum.each(0..(context.operations - 1), fn operation ->
      index = rem(context.round * 997 + operation * 37, embedding_count)
      id = elem(context.id_tuple, index)
      {:ok, embedding} = Collection.get(context.flat, id)
      churn_embedding!(context.flat, context.hnsw, embedding, operation)
    end)

    context.operations
  end

  @spec churn_embedding!(Collection.t(), Collection.t(), Embedding.t(), non_neg_integer()) :: :ok
  defp churn_embedding!(flat, hnsw, embedding, operation) when rem(operation, 2) == 0 do
    :ok = Collection.delete(flat, embedding.id)
    :ok = Collection.delete(hnsw, embedding.id)
    :ok = Collection.put(hnsw, embedding)
    :ok = Collection.put(flat, embedding)
  end

  defp churn_embedding!(flat, hnsw, embedding, _operation) do
    :ok = Collection.delete(hnsw, embedding.id)
    :ok = Collection.delete(flat, embedding.id)
    :ok = Collection.put(flat, embedding)
    :ok = Collection.put(hnsw, embedding)
  end

  @spec maybe_start_snapshot_task(map(), pos_integer()) :: Task.t() | nil
  defp maybe_start_snapshot_task(context, round) do
    every = context.config.snapshot_every_rounds

    if every > 0 and rem(round, every) == 0 do
      snapshot_context = %{
        flat: context.flat,
        hnsw: context.hnsw,
        query_tuple: context.query_tuple,
        limit: context.limit
      }

      Task.async(fn -> snapshot_cycle!(snapshot_context, round) end)
    end
  end

  @spec snapshot_cycle!(map(), pos_integer()) :: :ok
  defp snapshot_cycle!(context, round) do
    {source, target_index} =
      if rem(round, 2) == 0,
        do: {context.flat, :hnsw},
        else: {context.hnsw, :flat}

    unique = System.unique_integer([:positive, :monotonic])
    path = Path.join(System.tmp_dir!(), "vettore-endurance-#{unique}.ets")

    try do
      :ok = Collection.snapshot(source, path)

      load_options =
        case target_index do
          :flat -> [name: "snapshot-flat-#{unique}", index: :flat, index_options: []]
          :hnsw -> [name: "snapshot-hnsw-#{unique}", index: :hnsw, index_options: @hnsw_options]
        end

      {:ok, loaded} = Collection.load_snapshot(path, load_options)

      try do
        query = elem(context.query_tuple, rem(round, tuple_size(context.query_tuple)))
        {_elapsed_us, results} = timed_search!(loaded, query, context.limit)
        validate_results!(results, context.limit)
      after
        :ok = Collection.close(loaded)
      end
    after
      remove_snapshot(path)
    end
  end

  @spec await_optional_task(Task.t() | nil) :: 0 | 1
  defp await_optional_task(nil), do: 0

  defp await_optional_task(task) do
    :ok = Task.await(task, :infinity)
    1
  end

  @spec audit_collections!(map()) :: float()
  defp audit_collections!(context) do
    assert stored_ids!(context.flat) == context.expected_ids
    assert stored_ids!(context.hnsw) == context.expected_ids
    assert_memory_limit!(context.config.maximum_memory_bytes)

    recalls =
      context.query_tuple
      |> Tuple.to_list()
      |> Enum.take(context.config.audit_queries)
      |> Enum.map(fn query ->
        {_flat_us, exact_results} = timed_search!(context.flat, query, context.limit)
        {_hnsw_us, hnsw_results} = timed_search!(context.hnsw, query, context.limit)
        result_overlap(exact_results, hnsw_results)
      end)

    minimum_recall = Enum.min(recalls)

    assert minimum_recall >= context.config.minimum_recall,
           "HNSW audit recall #{minimum_recall} fell below #{context.config.minimum_recall}"

    minimum_recall
  end

  @spec stored_ids!(Collection.t()) :: MapSet.t(String.t())
  defp stored_ids!(collection) do
    {:ok, embeddings} = Collection.all(collection)
    ids = Enum.map(embeddings, & &1.id)

    assert length(ids) == MapSet.size(MapSet.new(ids))
    MapSet.new(ids)
  end

  @spec empty_reader_report() :: map()
  defp empty_reader_report do
    %{
      search_pairs: 0,
      flat_search_us: 0,
      hnsw_search_us: 0
    }
  end

  @spec empty_soak_round() :: map()
  defp empty_soak_round do
    %{
      search_pairs: 0,
      mutations: 0,
      snapshots: 0,
      flat_search_us: 0,
      hnsw_search_us: 0
    }
  end

  @spec merge_soak_round(map(), map()) :: map()
  defp merge_soak_round(stats, round_stats) do
    %{
      stats
      | rounds: stats.rounds + 1,
        search_pairs: stats.search_pairs + round_stats.search_pairs,
        mutations: stats.mutations + round_stats.mutations,
        snapshots: stats.snapshots + round_stats.snapshots,
        flat_search_us: stats.flat_search_us + round_stats.flat_search_us,
        hnsw_search_us: stats.hnsw_search_us + round_stats.hnsw_search_us
    }
  end

  @spec merge_audit_recall(map(), float() | nil) :: map()
  defp merge_audit_recall(stats, nil), do: stats

  defp merge_audit_recall(stats, recall) do
    %{stats | minimum_audit_recall: min(stats.minimum_audit_recall, recall)}
  end

  @spec maybe_print_soak_report(map(), pos_integer()) :: map()
  defp maybe_print_soak_report(stats, report_every_ms) do
    now_ms = monotonic_ms()

    if now_ms - stats.last_report_ms >= report_every_ms do
      print_soak_report(stats, false)
      %{stats | last_report_ms: now_ms}
    else
      stats
    end
  end

  @spec print_soak_report(map(), boolean()) :: :ok
  defp print_soak_report(stats, final?) do
    elapsed_seconds = elapsed_seconds(stats.started_ms)
    label = if final?, do: "SOAK FINAL", else: "SOAK"
    pairs = max(stats.search_pairs, 1)

    IO.puts(
      "#{label} elapsed=#{format_seconds(elapsed_seconds)} rounds=#{stats.rounds} " <>
        "search_pairs=#{stats.search_pairs} mutations=#{stats.mutations} " <>
        "snapshots=#{stats.snapshots} pairs/s=#{format_float(stats.search_pairs / elapsed_seconds)} " <>
        "flat_avg_us=#{format_float(stats.flat_search_us / pairs)} " <>
        "hnsw_avg_us=#{format_float(stats.hnsw_search_us / pairs)} " <>
        "audit_min_recall=#{format_float(stats.minimum_audit_recall)} " <>
        "memory_mb=#{format_float(memory_megabytes())}"
    )
  end

  @spec run_benchmark(map(), integer(), map(), pos_integer()) :: map()
  defp run_benchmark(context, deadline_ms, stats, round) do
    continue_benchmark(
      monotonic_ms() < deadline_ms,
      context,
      deadline_ms,
      stats,
      round
    )
  end

  @spec continue_benchmark(boolean(), map(), integer(), map(), pos_integer()) :: map()
  defp continue_benchmark(false, _context, _deadline_ms, stats, _round), do: stats

  defp continue_benchmark(true, context, deadline_ms, stats, round) do
    reports =
      1..context.config.workers
      |> Enum.map(fn worker ->
        Task.async(fn -> run_benchmark_worker(context, round, worker) end)
      end)
      |> Task.await_many(:infinity)

    scenarios = Enum.reduce(reports, stats.scenarios, &merge_scenario_stats/2)
    recall = maybe_audit_search_modes(context, stats.recall, round)

    assert_memory_limit!(context.config.maximum_memory_bytes)

    stats =
      %{stats | rounds: stats.rounds + 1, scenarios: scenarios, recall: recall}
      |> maybe_print_benchmark_report(context.config.report_every_ms)

    run_benchmark(context, deadline_ms, stats, round + 1)
  end

  @spec maybe_audit_search_modes(map(), map(), pos_integer()) :: map()
  defp maybe_audit_search_modes(context, previous_recall, round) do
    if rem(round, context.config.audit_every_rounds) == 0,
      do: audit_search_modes!(context),
      else: previous_recall
  end

  @spec run_benchmark_worker(map(), pos_integer(), pos_integer()) :: map()
  defp run_benchmark_worker(context, round, worker) do
    query_count = tuple_size(context.query_tuple)
    scenario_count = length(@benchmark_scenarios)

    Enum.reduce(0..(context.config.operations_per_round - 1), empty_scenario_stats(), fn
      operation, stats ->
        sequence = round * 10_007 + worker * 1_009 + operation
        scenario = Enum.at(@benchmark_scenarios, rem(sequence, scenario_count))
        query = elem(context.query_tuple, rem(sequence, query_count))

        {elapsed_us, results} = timed_scenario!(scenario, context, query)
        validate_results!(results, context.limit)
        record_scenario_latency(stats, scenario, elapsed_us)
    end)
  end

  @spec timed_scenario!(atom(), map(), [float()]) :: {non_neg_integer(), [Result.t()]}
  defp timed_scenario!(:flat, context, query),
    do: timed_search!(context.flat, query, context.limit)

  defp timed_scenario!(:hnsw, context, query),
    do: timed_search!(context.hnsw, query, context.limit)

  defp timed_scenario!(:funnel, context, query) do
    stages =
      [max(div(context.dimensions, 4), 1), max(div(context.dimensions, 2), 1), context.dimensions]
      |> Enum.uniq()

    timed_call!(fn ->
      Collection.funnel_search(context.flat, query,
        stages: stages,
        candidates: context.candidates,
        limit: context.limit
      )
    end)
  end

  defp timed_scenario!(:quantized, context, query) do
    timed_call!(fn ->
      Collection.quantized_search(context.flat, query,
        candidates: context.candidates,
        limit: context.limit
      )
    end)
  end

  defp timed_scenario!(:hybrid, context, query) do
    timed_call!(fn ->
      Collection.hybrid_search(context.hnsw, query,
        generators: [
          hnsw: [candidates: context.candidates],
          quantized: [candidates: context.candidates]
        ],
        rerank: :exact,
        limit: context.limit
      )
    end)
  end

  @spec audit_search_modes!(map()) :: map()
  defp audit_search_modes!(context) do
    recall =
      context.query_tuple
      |> Tuple.to_list()
      |> Enum.take(context.config.audit_queries)
      |> Enum.reduce(empty_recall_stats(), fn query, stats ->
        {_elapsed_us, exact_results} = timed_scenario!(:flat, context, query)

        Enum.reduce(@benchmark_scenarios, stats, fn scenario, scenario_stats ->
          {_scenario_us, results} = timed_scenario!(scenario, context, query)
          overlap = result_overlap(exact_results, results)
          record_recall(scenario_stats, scenario, overlap)
        end)
      end)
      |> finalize_recall_stats()

    hnsw_recall = Map.fetch!(recall, :hnsw)

    assert hnsw_recall.minimum >= context.config.minimum_hnsw_recall,
           "HNSW recall #{hnsw_recall.minimum} fell below " <>
             "#{context.config.minimum_hnsw_recall}"

    recall
  end

  @spec empty_scenario_stats() :: map()
  defp empty_scenario_stats do
    Map.new(@benchmark_scenarios, &{&1, empty_latency_stats()})
  end

  @spec empty_latency_stats() :: latency_stats()
  defp empty_latency_stats do
    %{count: 0, total_us: 0, maximum_us: 0, buckets: %{}}
  end

  @spec record_scenario_latency(map(), atom(), non_neg_integer()) :: map()
  defp record_scenario_latency(stats, scenario, elapsed_us) do
    Map.update!(stats, scenario, &record_latency(&1, elapsed_us))
  end

  @spec record_latency(latency_stats(), non_neg_integer()) :: latency_stats()
  defp record_latency(stats, elapsed_us) do
    bucket = Enum.find(@latency_buckets_us, :infinity, &(&1 >= elapsed_us))

    %{
      count: stats.count + 1,
      total_us: stats.total_us + elapsed_us,
      maximum_us: max(stats.maximum_us, elapsed_us),
      buckets: Map.update(stats.buckets, bucket, 1, &(&1 + 1))
    }
  end

  @spec merge_scenario_stats(map(), map()) :: map()
  defp merge_scenario_stats(report, accumulated) do
    Map.merge(accumulated, report, fn _scenario, left, right ->
      %{
        count: left.count + right.count,
        total_us: left.total_us + right.total_us,
        maximum_us: max(left.maximum_us, right.maximum_us),
        buckets: Map.merge(left.buckets, right.buckets, fn _bucket, a, b -> a + b end)
      }
    end)
  end

  @spec empty_recall_stats() :: map()
  defp empty_recall_stats do
    Map.new(@benchmark_scenarios, &{&1, %{count: 0, total: 0.0, minimum: 1.0}})
  end

  @spec record_recall(map(), atom(), float()) :: map()
  defp record_recall(stats, scenario, recall) do
    Map.update!(stats, scenario, fn current ->
      %{
        count: current.count + 1,
        total: current.total + recall,
        minimum: min(current.minimum, recall)
      }
    end)
  end

  @spec finalize_recall_stats(map()) :: map()
  defp finalize_recall_stats(stats) do
    Map.new(stats, fn {scenario, values} ->
      {scenario, %{average: values.total / max(values.count, 1), minimum: values.minimum}}
    end)
  end

  @spec maybe_print_benchmark_report(map(), pos_integer()) :: map()
  defp maybe_print_benchmark_report(stats, report_every_ms) do
    now_ms = monotonic_ms()

    if now_ms - stats.last_report_ms >= report_every_ms do
      print_benchmark_report(stats, false)
      %{stats | last_report_ms: now_ms}
    else
      stats
    end
  end

  @spec print_benchmark_report(map(), boolean()) :: :ok
  defp print_benchmark_report(stats, final?) do
    elapsed = elapsed_seconds(stats.started_ms)
    label = if final?, do: "ENDURANCE BENCHMARK FINAL", else: "ENDURANCE BENCHMARK"

    IO.puts(
      "\n#{label} elapsed=#{format_seconds(elapsed)} rounds=#{stats.rounds} " <>
        "memory_mb=#{format_float(memory_megabytes())}"
    )

    IO.puts("scenario    operations  ops/s      avg_us   p50_us   p95_us   p99_us   max_us")

    Enum.each(@benchmark_scenarios, fn scenario ->
      latency = Map.fetch!(stats.scenarios, scenario)
      throughput = latency.count / elapsed
      average = latency.total_us / max(latency.count, 1)

      IO.puts(
        String.pad_trailing(Atom.to_string(scenario), 12) <>
          String.pad_leading(Integer.to_string(latency.count), 10) <>
          String.pad_leading(format_float(throughput), 11) <>
          String.pad_leading(format_float(average), 10) <>
          String.pad_leading(Integer.to_string(percentile_us(latency, 50)), 9) <>
          String.pad_leading(Integer.to_string(percentile_us(latency, 95)), 9) <>
          String.pad_leading(Integer.to_string(percentile_us(latency, 99)), 9) <>
          String.pad_leading(Integer.to_string(latency.maximum_us), 9)
      )
    end)

    print_recall(stats.recall)
  end

  @spec print_recall(map()) :: :ok
  defp print_recall(recall) when map_size(recall) == 0 do
    IO.puts("recall audit pending")
  end

  defp print_recall(recall) do
    summary =
      Enum.map_join(@benchmark_scenarios, " ", fn scenario ->
        values = Map.fetch!(recall, scenario)

        "#{scenario}=avg:#{format_float(values.average)},min:#{format_float(values.minimum)}"
      end)

    IO.puts("recall@k #{summary}")
  end

  @spec percentile_us(latency_stats(), pos_integer()) :: non_neg_integer()
  defp percentile_us(%{count: 0}, _percentile), do: 0

  defp percentile_us(stats, percentile) do
    target = ceil(stats.count * percentile / 100)

    (@latency_buckets_us ++ [:infinity])
    |> Enum.reduce_while(0, fn bucket, cumulative ->
      percentile_step(bucket, cumulative, target, stats)
    end)
  end

  @spec percentile_step(
          pos_integer() | :infinity,
          non_neg_integer(),
          pos_integer(),
          latency_stats()
        ) :: {:cont, non_neg_integer()} | {:halt, non_neg_integer()}
  defp percentile_step(bucket, cumulative, target, stats) do
    cumulative = cumulative + Map.get(stats.buckets, bucket, 0)
    percentile_result(bucket, cumulative, target, stats.maximum_us)
  end

  @spec percentile_result(
          pos_integer() | :infinity,
          non_neg_integer(),
          pos_integer(),
          non_neg_integer()
        ) :: {:cont, non_neg_integer()} | {:halt, non_neg_integer()}
  defp percentile_result(bucket, cumulative, target, maximum_us) when cumulative >= target,
    do: {:halt, latency_bucket_value(bucket, maximum_us)}

  defp percentile_result(_bucket, cumulative, _target, _maximum_us),
    do: {:cont, cumulative}

  @spec latency_bucket_value(pos_integer() | :infinity, non_neg_integer()) :: non_neg_integer()
  defp latency_bucket_value(:infinity, maximum_us), do: maximum_us
  defp latency_bucket_value(bucket, _maximum_us), do: bucket

  @spec total_benchmark_operations(map()) :: non_neg_integer()
  defp total_benchmark_operations(scenarios) do
    Enum.reduce(scenarios, 0, fn {_scenario, stats}, total -> total + stats.count end)
  end

  @spec timed_search!(Collection.t(), [float()], pos_integer()) ::
          {non_neg_integer(), [Result.t()]}
  defp timed_search!(collection, query, limit) do
    timed_call!(fn -> Collection.search(collection, query, limit: limit) end)
  end

  @spec timed_call!((-> {:ok, [Result.t()]} | {:error, term()})) ::
          {non_neg_integer(), [Result.t()]}
  defp timed_call!(fun) do
    {elapsed_us, result} = :timer.tc(fun)

    case result do
      {:ok, results} when is_list(results) -> {elapsed_us, results}
      {:error, reason} -> raise "endurance operation failed: #{inspect(reason)}"
    end
  end

  @spec validate_results!([Result.t()], pos_integer()) :: :ok
  defp validate_results!(results, limit) do
    assert length(results) <= limit
    assert MapSet.size(MapSet.new(results, & &1.id)) == length(results)

    assert Enum.all?(results, fn
             %Result{id: id, score: score, distance: distance}
             when is_binary(id) and is_number(score) and is_number(distance) ->
               finite_number?(score) and finite_number?(distance)

             _result ->
               false
           end)

    :ok
  end

  @spec result_overlap([Result.t()], [Result.t()]) :: float()
  defp result_overlap(reference, results) do
    reference_ids = MapSet.new(reference, & &1.id)
    result_ids = MapSet.new(results, & &1.id)
    denominator = max(MapSet.size(reference_ids), 1)

    reference_ids
    |> MapSet.intersection(result_ids)
    |> MapSet.size()
    |> Kernel./(denominator)
  end

  @spec finite_number?(number()) :: boolean()
  defp finite_number?(value), do: value >= -@f32_max and value <= @f32_max

  @spec assert_memory_limit!(non_neg_integer()) :: :ok
  defp assert_memory_limit!(0), do: :ok

  defp assert_memory_limit!(maximum_bytes) do
    current_bytes = :erlang.memory(:total)

    assert current_bytes <= maximum_bytes,
           "BEAM memory #{current_bytes} exceeded configured limit #{maximum_bytes}"

    :ok
  end

  @spec remove_snapshot(Path.t()) :: :ok
  defp remove_snapshot(path) do
    case File.rm(path) do
      :ok -> :ok
      {:error, :enoent} -> :ok
      {:error, reason} -> raise "could not remove endurance snapshot: #{inspect(reason)}"
    end
  end

  @spec ensure_limit_fits!(pos_integer(), pos_integer(), String.t()) :: :ok
  defp ensure_limit_fits!(limit, upper_bound, name) do
    if limit <= upper_bound,
      do: :ok,
      else: raise(ArgumentError, "#{name} must not exceed #{upper_bound}")
  end

  @spec positive_integer_env!(String.t(), pos_integer()) :: pos_integer()
  defp positive_integer_env!(name, default) do
    case Integer.parse(System.get_env(name, Integer.to_string(default))) do
      {value, ""} when value > 0 -> value
      _other -> raise ArgumentError, "#{name} must be a positive integer"
    end
  end

  @spec non_negative_integer_env!(String.t(), non_neg_integer()) :: non_neg_integer()
  defp non_negative_integer_env!(name, default) do
    case Integer.parse(System.get_env(name, Integer.to_string(default))) do
      {value, ""} when value >= 0 -> value
      _other -> raise ArgumentError, "#{name} must be a non-negative integer"
    end
  end

  @spec probability_env!(String.t(), float()) :: float()
  defp probability_env!(name, default) do
    case Float.parse(System.get_env(name, Float.to_string(default))) do
      {value, ""} when value >= 0.0 and value <= 1.0 -> value
      _other -> raise ArgumentError, "#{name} must be between 0.0 and 1.0"
    end
  end

  @spec monotonic_ms() :: integer()
  defp monotonic_ms, do: System.monotonic_time(:millisecond)

  @spec elapsed_seconds(integer()) :: float()
  defp elapsed_seconds(started_ms), do: max((monotonic_ms() - started_ms) / 1_000, 0.001)

  @spec memory_megabytes() :: float()
  defp memory_megabytes, do: :erlang.memory(:total) / (1_024 * 1_024)

  @spec format_seconds(float()) :: String.t()
  defp format_seconds(seconds) do
    total = trunc(seconds)
    hours = div(total, 3_600)
    minutes = div(rem(total, 3_600), 60)
    remaining_seconds = rem(total, 60)

    :io_lib.format("~2..0B:~2..0B:~2..0B", [hours, minutes, remaining_seconds])
    |> IO.iodata_to_binary()
  end

  @spec format_float(number()) :: String.t()
  defp format_float(value), do: :erlang.float_to_binary(value / 1, decimals: 2)
end
