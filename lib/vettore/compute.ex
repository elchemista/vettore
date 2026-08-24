defmodule Vettore.Compute do
  @moduledoc """
  Selects native CPU or GPU execution for dense-vector primitives and exact
  batched search.

  CPU execution uses Vettore's Rust SIMD kernels and remains the default. GPU
  execution is provided directly by the native Rust NIF through `wgpu`; Nx is
  not involved and remains an optional interchange format only.

  GPU execution can be enabled globally:

      config :vettore,
        gpu: true,
        gpu_fallback: :cpu

  Or per call with `gpu: true`. For forced GPU execution, `gpu_fallback`
  controls what happens when an adapter cannot be initialized or execution
  fails:

    * `:cpu` (default) transparently uses the SIMD CPU kernel;
    * `:error` returns a stable tagged error.

  `gpu: :auto` uses the GPU only when one is available and the operation has at
  least `:gpu_min_size` coordinates (1,000,000 by default). For Flat search the
  workload is the resident matrix shape (`rows * dimensions`). It always uses CPU
  when no adapter is available, independently of the forced-GPU fallback policy.
  This avoids paying GPU transfer and synchronization costs for small vectors.
  """

  require Logger

  alias Vettore.Nifs

  @default_gpu_min_size 1_000_000

  @type device :: :cpu | :gpu

  @doc "Returns `true` when Vettore can initialize a hardware GPU through wgpu."
  @spec gpu_detected?() :: boolean()
  def gpu_detected?, do: gpu_detected?(&Nifs.gpu_detected/0)

  @doc false
  @spec gpu_detected?((-> term())) :: boolean()
  def gpu_detected?(probe) when is_function(probe, 0) do
    probe.() == true
  rescue
    exception ->
      log_gpu_failure("detection", :error, exception)
      false
  catch
    kind, reason ->
      log_gpu_failure("detection", kind, reason)
      false
  end

  @doc "Compatibility alias for `gpu_detected?/0`."
  @spec gpu_detected() :: boolean()
  def gpu_detected, do: gpu_detected?()

  @doc "Returns information about the selected hardware GPU adapter."
  @spec gpu_info() ::
          {:ok, %{name: String.t(), backend: String.t(), device_type: String.t()}}
          | {:error, :gpu_not_available | term()}
  def gpu_info, do: gpu_info(&Nifs.gpu_info/0)

  @doc false
  @spec gpu_info((-> term())) ::
          {:ok, %{name: String.t(), backend: String.t(), device_type: String.t()}}
          | {:error, :gpu_not_available | term()}
  def gpu_info(provider) when is_function(provider, 0) do
    case provider.() do
      {:ok, {name, backend, device_type}} ->
        {:ok, %{name: name, backend: backend, device_type: device_type}}

      {:error, "gpu not detected"} ->
        {:error, :gpu_not_available}

      {:error, "gpu " <> _reason} ->
        {:error, :gpu_failed}

      {:error, reason} ->
        {:error, reason}
    end
  rescue
    exception ->
      log_gpu_failure("adapter info", :error, exception)
      {:error, :gpu_not_available}
  catch
    kind, reason ->
      log_gpu_failure("adapter info", kind, reason)
      {:error, :gpu_not_available}
  end

  @doc "Returns the configured mode, fallback, threshold, and detected GPU details."
  @spec info() :: map()
  def info, do: info(&gpu_detected?/0, &gpu_info/0)

  @doc false
  @spec info((-> boolean()), (-> term())) :: map()
  def info(detected, adapter) when is_function(detected, 0) and is_function(adapter, 0) do
    %{
      gpu: Application.get_env(:vettore, :gpu, false),
      fallback: Application.get_env(:vettore, :gpu_fallback, :cpu),
      min_size: Application.get_env(:vettore, :gpu_min_size, @default_gpu_min_size),
      detected?: detected.(),
      adapter: adapter_info(adapter)
    }
  end

  @doc false
  @spec device(keyword(), non_neg_integer()) :: {:ok, device()} | {:error, term()}
  def device(opts, workload_size)
      when is_list(opts) and is_integer(workload_size) and workload_size >= 0 do
    with :ok <- validate_options(opts),
         {:ok, selection} <- gpu_selection(opts),
         {:ok, fallback} <- gpu_fallback(opts),
         {:ok, min_size} <- gpu_min_size(opts) do
      detected? =
        if gpu_probe_required?(selection, min_size, workload_size),
          do: gpu_detected?(),
          else: false

      select_device(selection, fallback, min_size, workload_size, detected?)
    end
  end

  def device(_opts, _workload_size), do: {:error, :invalid_options}

  @doc false
  @spec run(keyword(), non_neg_integer(), (-> term()), (-> term())) :: term()
  def run(opts, workload_size, cpu, gpu)
      when is_function(cpu, 0) and is_function(gpu, 0) do
    with {:ok, device} <- device(opts, workload_size) do
      {:ok, fallback} = gpu_fallback(opts)
      run_device(device, fallback, cpu, gpu)
    end
  end

  @doc false
  @spec normalize_search_result(term()) :: term()
  def normalize_search_result({:error, "gpu not detected"}),
    do: {:error, :gpu_not_available}

  def normalize_search_result({:error, "gpu flat top-k supports" <> _reason}),
    do: {:error, :gpu_limit_too_large}

  def normalize_search_result({:error, "gpu " <> _reason}), do: {:error, :gpu_failed}
  def normalize_search_result(result), do: result

  @doc false
  @spec select_device(
          boolean() | :auto,
          :cpu | :error,
          pos_integer(),
          non_neg_integer(),
          boolean()
        ) :: {:ok, device()} | {:error, :gpu_not_available}
  def select_device(false, _fallback, _min_size, _workload_size, _detected?), do: {:ok, :cpu}

  def select_device(:auto, _fallback, min_size, workload_size, _detected?)
      when workload_size < min_size,
      do: {:ok, :cpu}

  def select_device(:auto, _fallback, _min_size, _workload_size, false), do: {:ok, :cpu}

  def select_device(selection, fallback, _min_size, _workload_size, detected?)
      when selection in [true, :auto] do
    cond do
      detected? -> {:ok, :gpu}
      fallback == :cpu -> {:ok, :cpu}
      true -> {:error, :gpu_not_available}
    end
  end

  defp gpu_probe_required?(false, _min_size, _workload_size), do: false

  defp gpu_probe_required?(:auto, min_size, workload_size), do: workload_size >= min_size

  defp gpu_probe_required?(true, _min_size, _workload_size), do: true

  @doc false
  @spec run_device(device(), :cpu | :error, (-> term()), (-> term())) :: term()
  def run_device(:cpu, _fallback, cpu, _gpu), do: cpu.()

  def run_device(:gpu, fallback, cpu, gpu) do
    case safely_run(gpu) do
      {:error, _reason} = error ->
        case fallback do
          :cpu -> cpu.()
          :error -> error
        end

      result ->
        result
    end
  end

  defp safely_run(operation) do
    operation.()
  rescue
    exception ->
      log_gpu_failure("operation", :error, exception)
      {:error, :gpu_failed}
  catch
    kind, reason ->
      log_gpu_failure("operation", kind, reason)
      {:error, :gpu_failed}
  end

  @doc false
  @spec validate_options(term()) :: :ok | {:error, term()}
  def validate_options(opts) do
    with :ok <- validate_option_keys(opts),
         {:ok, _selection} <- gpu_selection(opts),
         {:ok, _fallback} <- gpu_fallback(opts),
         {:ok, _min_size} <- gpu_min_size(opts) do
      :ok
    end
  end

  @doc false
  @spec selection(keyword()) :: {:ok, boolean() | :auto} | {:error, term()}
  def selection(opts) do
    with :ok <- validate_options(opts) do
      gpu_selection(opts)
    end
  end

  defp validate_option_keys(opts) do
    allowed = [:gpu, :gpu_fallback, :gpu_min_size]

    if Keyword.keyword?(opts) do
      keys = Keyword.keys(opts)

      if keys == Enum.uniq(keys) and Enum.all?(keys, &(&1 in allowed)),
        do: :ok,
        else: {:error, :invalid_options}
    else
      {:error, :invalid_options}
    end
  end

  defp gpu_selection(opts) do
    case Keyword.get(opts, :gpu, Application.get_env(:vettore, :gpu, false)) do
      selection when selection in [true, false, :auto] -> {:ok, selection}
      _selection -> {:error, :invalid_gpu_option}
    end
  end

  defp gpu_fallback(opts) do
    case Keyword.get(opts, :gpu_fallback, Application.get_env(:vettore, :gpu_fallback, :cpu)) do
      fallback when fallback in [:cpu, :error] -> {:ok, fallback}
      _fallback -> {:error, :invalid_gpu_fallback}
    end
  end

  defp gpu_min_size(opts) do
    case Keyword.get(
           opts,
           :gpu_min_size,
           Application.get_env(:vettore, :gpu_min_size, @default_gpu_min_size)
         ) do
      min_size when is_integer(min_size) and min_size > 0 -> {:ok, min_size}
      _min_size -> {:error, :invalid_gpu_min_size}
    end
  end

  defp adapter_info(provider) do
    case provider.() do
      {:ok, info} -> info
      {:error, _reason} -> nil
    end
  end

  defp log_gpu_failure(operation, kind, reason) do
    Logger.warning(fn ->
      formatted = inspect(reason, limit: 10, printable_limit: 200)
      "Vettore GPU #{operation} failed (#{kind}): #{formatted}"
    end)
  end
end
