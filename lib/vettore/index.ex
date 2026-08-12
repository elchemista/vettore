defmodule Vettore.Index do
  @moduledoc """
  Search index behaviour.

  Indexes may keep acceleration state, but ETS remains the canonical record
  store. Implementations must return `Vettore.Result` structs.
  """

  alias Vettore.{Distance, Embedding, Identifier, Result}

  @type context :: map()

  @max_nif_usize 4_294_967_295
  @f32_max 3.402_823_466_385_288_6e38

  @callback new(Distance.metric(), keyword()) :: {:ok, term()} | {:error, term()}
  @callback put(context(), Embedding.t()) :: :ok | {:error, term()}
  @callback put_many(context(), [Embedding.t()]) :: :ok | {:error, term()}
  @callback delete(context(), String.t()) :: :ok | {:error, term()}
  @callback search(context(), [number()], keyword()) ::
              {:ok, [Result.t()]} | {:error, term()}
  @callback close(context()) :: :ok | {:error, term()}

  @optional_callbacks close: 1

  @doc false
  @spec prepare_query(context(), term()) :: {:ok, [float()]} | {:error, term()}
  def prepare_query(%{dimensions: dimensions, normalize: normalize} = context, query) do
    with :ok <- ensure_open(context),
         :ok <- validate_vector(query, dimensions) do
      Distance.normalize(query, normalize)
    end
  end

  @doc false
  @spec ensure_open(context()) :: :ok | {:error, :closed}
  def ensure_open(%{store_mod: store_mod, store_state: store_state}) when is_atom(store_mod) do
    if function_exported?(store_mod, :alive?, 1) and not store_mod.alive?(store_state) do
      {:error, :closed}
    else
      :ok
    end
  end

  def ensure_open(_context), do: {:error, :closed}

  @doc false
  @spec hydrate_results(context(), [{String.t(), number()}]) :: [Result.t()]
  def hydrate_results(context, hits), do: Enum.flat_map(hits, &hydrate_result(context, &1))

  @doc false
  @spec normalize_write_result(:ok | {:ok, {}} | {:error, reason}) ::
          :ok | {:error, reason}
        when reason: term()
  def normalize_write_result({:ok, {}}), do: :ok
  def normalize_write_result(:ok), do: :ok
  def normalize_write_result({:error, _reason} = error), do: error

  @doc false
  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  def validate_limit(limit)
      when is_integer(limit) and limit > 0 and limit <= @max_nif_usize,
      do: :ok

  def validate_limit(_limit), do: {:error, :invalid_limit}

  @doc false
  @spec validate_search_options(term()) :: :ok | {:error, :invalid_search_options}
  def validate_search_options(opts) when is_list(opts) do
    if Keyword.keyword?(opts) and Enum.all?(Keyword.keys(opts), &(&1 == :limit)),
      do: :ok,
      else: {:error, :invalid_search_options}
  end

  def validate_search_options(_opts), do: {:error, :invalid_search_options}

  @spec hydrate_result(context(), {String.t(), number()}) :: [Result.t()]
  defp hydrate_result(%{store_mod: store_mod} = context, {id, raw}) when is_atom(store_mod) do
    with :ok <- Identifier.validate_utf8(id),
         {:ok, %Embedding{} = embedding} <- store_mod.get(context.store_state, id) do
      {score, distance} = Distance.result_values(context.metric, raw, context.score)

      [
        %Result{
          id: id,
          value: embedding.value,
          score: score,
          distance: distance,
          metric: context.metric,
          metadata: embedding.metadata
        }
      ]
    else
      {:error, _reason} -> []
    end
  end

  defp hydrate_result(_context, _hit), do: []

  @spec validate_vector(term(), term()) ::
          :ok | {:error, :dimension_mismatch | :invalid_vector}
  defp validate_vector(vector, dimensions) when is_list(vector) do
    cond do
      length(vector) != dimensions -> {:error, :dimension_mismatch}
      Enum.all?(vector, &finite_f32?/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_vector(_vector, _dimensions), do: {:error, :invalid_vector}

  @spec finite_f32?(term()) :: boolean()
  defp finite_f32?(value) when is_integer(value), do: value >= -@f32_max and value <= @f32_max
  defp finite_f32?(value) when is_float(value), do: value >= -@f32_max and value <= @f32_max
  defp finite_f32?(_value), do: false
end
