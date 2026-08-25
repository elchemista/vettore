defmodule VettoreGpuComputeTest do
  use ExUnit.Case, async: false

  alias Vettore.Compute
  alias Vettore.Vector

  @application_keys [:gpu, :gpu_fallback, :gpu_min_size]

  setup do
    previous = Map.new(@application_keys, &{&1, Application.get_env(:vettore, &1, :missing)})

    on_exit(fn ->
      Enum.each(previous, fn
        {key, :missing} -> Application.delete_env(:vettore, key)
        {key, value} -> Application.put_env(:vettore, key, value)
      end)
    end)

    :ok
  end

  describe "GPU detection and configuration" do
    test "CI can require a real hardware or software adapter" do
      if System.get_env("VETTORE_REQUIRE_GPU") == "1" do
        assert Compute.gpu_detected?(),
               "VETTORE_REQUIRE_GPU=1 but wgpu could not initialize an adapter"
      end
    end

    test "detection is boolean and adapter diagnostics agree with it" do
      assert is_boolean(Vettore.gpu_detected?())
      assert Vettore.gpu_detected() == Vettore.gpu_detected?()
      assert Compute.gpu_detected() == Compute.gpu_detected?()

      if Compute.gpu_detected?() do
        assert {:ok, %{name: name, backend: backend, device_type: device_type}} =
                 Vettore.gpu_info()

        assert is_binary(name) and name != ""
        assert backend in ["vulkan", "metal", "dx12", "gl", "browser_webgpu"]
        assert device_type in ["discrete_gpu", "integrated_gpu", "virtual_gpu", "other", "cpu"]
      else
        assert {:error, _reason} = Vettore.gpu_info()
      end

      refute Compute.gpu_detected?(fn -> raise "probe failure" end)
      refute Compute.gpu_detected?(fn -> throw(:probe_failure) end)

      assert {:error, :gpu_not_available} =
               Compute.gpu_info(fn -> {:error, "gpu not detected"} end)

      assert {:error, :driver_failure} = Compute.gpu_info(fn -> {:error, :driver_failure} end)

      assert {:error, :gpu_failed} =
               Compute.gpu_info(fn -> {:error, "gpu device initialization failed"} end)

      assert {:ok, %{name: "test adapter", backend: "vulkan", device_type: "cpu"}} =
               Compute.gpu_info(fn -> {:ok, {"test adapter", "vulkan", "cpu"}} end)

      assert {:error, :gpu_not_available} = Compute.gpu_info(fn -> raise "info failure" end)
      assert {:error, :gpu_not_available} = Compute.gpu_info(fn -> throw(:info_failure) end)

      fallback_info =
        Compute.info(fn -> false end, fn -> {:error, :gpu_not_available} end)

      refute fallback_info.detected?
      assert fallback_info.adapter == nil

      detected_info =
        Compute.info(fn -> true end, fn -> {:ok, %{name: "test adapter"}} end)

      assert detected_info.detected?
      assert detected_info.adapter == %{name: "test adapter"}
    end

    test "pure device selection covers forced, automatic, and fallback modes" do
      assert {:ok, :cpu} = Compute.select_device(false, :error, 100, 1_000, true)
      assert {:ok, :cpu} = Compute.select_device(:auto, :error, 100, 99, true)
      assert {:ok, :gpu} = Compute.select_device(:auto, :error, 100, 100, true)
      assert {:ok, :gpu} = Compute.select_device(true, :cpu, 100, 1, true)
      assert {:ok, :cpu} = Compute.select_device(true, :cpu, 100, 1, false)
      assert {:ok, :cpu} = Compute.select_device(:auto, :error, 100, 100, false)

      assert {:error, :gpu_not_available} =
               Compute.select_device(true, :error, 100, 1, false)
    end

    test "GPU operation failures obey the explicit fallback policy" do
      assert {:ok, :cpu} =
               Compute.run_device(:cpu, :error, fn -> {:ok, :cpu} end, fn -> flunk() end)

      assert {:ok, :gpu} =
               Compute.run_device(:gpu, :error, fn -> flunk() end, fn -> {:ok, :gpu} end)

      assert {:ok, :cpu} =
               Compute.run_device(
                 :gpu,
                 :cpu,
                 fn -> {:ok, :cpu} end,
                 fn -> {:error, :device_lost} end
               )

      assert {:error, :device_lost} =
               Compute.run_device(
                 :gpu,
                 :error,
                 fn -> flunk() end,
                 fn -> {:error, :device_lost} end
               )

      assert {:error, :gpu_failed} =
               Compute.run_device(:gpu, :error, fn -> flunk() end, fn -> raise "broken GPU" end)

      assert {:ok, :cpu} =
               Compute.run_device(:gpu, :cpu, fn -> {:ok, :cpu} end, fn -> throw(:broken) end)
    end

    test "global configuration and validation are deterministic" do
      Application.put_env(:vettore, :gpu, :auto)
      Application.put_env(:vettore, :gpu_fallback, :cpu)
      Application.put_env(:vettore, :gpu_min_size, 42)

      info = Compute.info()
      assert info.gpu == :auto
      assert info.fallback == :cpu
      assert info.min_size == 42
      assert info.detected? == Compute.gpu_detected?()

      assert {:ok, :cpu} = Compute.device([], 41)
      assert {:ok, :cpu} = Compute.device([gpu: false], 1_000)
      assert {:error, :invalid_options} = Compute.device([:bad], 1)
      assert {:error, :invalid_options} = Compute.device([gpu: false, gpu: true], 1)
      assert {:error, :invalid_options} = Compute.device([unknown: true], 1)
      assert {:error, :invalid_options} = Compute.device(:bad, 1)
      assert {:error, :invalid_options} = Compute.device([], -1)

      Application.put_env(:vettore, :gpu, :sometimes)
      assert {:error, :invalid_gpu_option} = Compute.device([], 100)
      Application.put_env(:vettore, :gpu, false)
      Application.put_env(:vettore, :gpu_fallback, :sometimes)
      assert {:error, :invalid_gpu_fallback} = Compute.device([], 100)
      Application.put_env(:vettore, :gpu_fallback, :cpu)
      Application.put_env(:vettore, :gpu_min_size, 0)
      assert {:error, :invalid_gpu_min_size} = Compute.device([], 100)
    end
  end

  describe "CPU/GPU vector API" do
    test "Distance primitives accept the same global and per-call GPU selection" do
      left = [1.0, 0.0, 2.0]
      right = [0.0, 3.0, 2.0]
      fallback = if Compute.gpu_detected?(), do: :error, else: :cpu

      for metric <- [
            :l2,
            :l2_squared,
            :inner_product,
            :negative_inner_product,
            :manhattan,
            :chebyshev,
            :hamming,
            :jaccard
          ] do
        assert {:ok, cpu} = apply(Vettore.Distance, metric, [left, right, [gpu: false]])

        assert {:ok, gpu} =
                 apply(Vettore.Distance, metric, [
                   left,
                   right,
                   [gpu: true, gpu_fallback: fallback]
                 ])

        assert_in_delta gpu, cpu, max(abs(cpu) * 1.0e-5, 1.0e-6)
      end

      for method <- [:none, :l2, :zscore, :minmax] do
        assert {:ok, cpu} = Vettore.Distance.normalize(left, method, gpu: false)

        assert {:ok, gpu} =
                 Vettore.Distance.normalize(left, method,
                   gpu: true,
                   gpu_fallback: fallback
                 )

        for {gpu_value, cpu_value} <- Enum.zip(gpu, cpu) do
          assert_in_delta gpu_value, cpu_value, 1.0e-5
        end
      end

      assert {:ok, cpu_cosine} = Vettore.Distance.cosine(left, right, gpu: false)

      assert {:ok, gpu_cosine} =
               Vettore.Distance.cosine(left, right, gpu: true, gpu_fallback: fallback)

      assert_in_delta gpu_cosine, cpu_cosine, 1.0e-5

      Application.put_env(:vettore, :gpu, true)
      Application.put_env(:vettore, :gpu_fallback, fallback)
      assert {:ok, global} = Vettore.Distance.l2(left, right)
      assert_in_delta global, 10.0 |> :math.sqrt(), 1.0e-5

      assert {:error, :invalid_options} =
               Vettore.Distance.l2(left, right, gpu: false, gpu: true)

      assert {:error, :invalid_options} = Vettore.Distance.normalize(left, :l2, :bad)
    end

    test "explicit CPU mode supports all metrics and named option arities" do
      left = [1.0, 0.0, 2.0]
      right = [0.0, 3.0, 2.0]

      calls = [
        l2: &Vector.l2/3,
        l2_squared: &Vector.l2_squared/3,
        inner_product: &Vector.inner_product/3,
        negative_inner_product: &Vector.negative_inner_product/3,
        manhattan: &Vector.manhattan/3,
        chebyshev: &Vector.chebyshev/3,
        hamming: &Vector.hamming/3,
        jaccard: &Vector.jaccard/3
      ]

      for {metric, function} <- calls do
        assert {:ok, expected} = Vector.metric(left, right, metric)
        assert {:ok, actual} = function.(left, right, gpu: false)
        assert_in_delta actual, expected, 1.0e-6
      end

      assert Vector.dot_product(left, right, gpu: false) ==
               Vector.inner_product(left, right, gpu: false)

      assert Vettore.Distance.euclidean(left, right, gpu: false) ==
               Vettore.Distance.l2(left, right, gpu: false)

      assert Vettore.Distance.dot_product(left, right, gpu: false) ==
               Vettore.Distance.inner_product(left, right, gpu: false)

      assert {:ok, cosine} = Vector.cosine(left, right, gpu: false)
      assert {:ok, expected_cosine} = Vector.metric(left, right, :cosine)
      assert_in_delta cosine, expected_cosine, 1.0e-6

      assert {:ok, transformed} =
               Vettore.Distance.cosine(left, right, normalize: :zscore, gpu: false)

      assert is_float(transformed)

      assert {:ok, minmax} =
               Vettore.Distance.cosine(left, right, normalize: :minmax, gpu: false)

      assert is_float(minmax)
    end

    test "automatic mode below its threshold stays on CPU" do
      assert {:ok, 5.0} =
               Vector.l2([0.0, 0.0], [3.0, 4.0], gpu: :auto, gpu_min_size: 10)

      assert {:ok, normalized} =
               Vector.normalize([3.0, 4.0], :l2, gpu: :auto, gpu_min_size: 10)

      assert_in_delta Enum.at(normalized, 0), 0.6, 1.0e-6
      assert_in_delta Enum.at(normalized, 1), 0.8, 1.0e-6

      assert {:ok, [2.0, 3.0]} =
               Vector.mean_pool([[1.0, 2.0], [3.0, 4.0]],
                 gpu: :auto,
                 gpu_min_size: 10
               )
    end

    test "forced GPU either executes real kernels or follows the selected fallback" do
      matrix = f32_binary([1.0, 2.0, 3.0, 4.0])

      assert {:ok, 5.0} =
               Vector.l2([0.0, 0.0], [3.0, 4.0], gpu: true, gpu_fallback: :cpu)

      assert {:ok, normalized} =
               Vector.normalize([3.0, 4.0], :l2, gpu: true, gpu_fallback: :cpu)

      assert_in_delta Enum.at(normalized, 0), 0.6, 1.0e-5
      assert_in_delta Enum.at(normalized, 1), 0.8, 1.0e-5

      assert {:ok, pooled} =
               Vector.mean_pool_f32(matrix, 2, [0, 1],
                 as: :list,
                 gpu: true,
                 gpu_fallback: :cpu
               )

      assert_in_delta Enum.at(pooled, 0), 2.0, 1.0e-5
      assert_in_delta Enum.at(pooled, 1), 3.0, 1.0e-5

      if Compute.gpu_detected?() do
        assert {:ok, 5.0} =
                 Vector.l2([0.0, 0.0], [3.0, 4.0], gpu: true, gpu_fallback: :error)

        assert_gpu_parity()
      else
        assert {:error, :gpu_not_available} =
                 Vector.l2([0.0], [1.0], gpu: true, gpu_fallback: :error)
      end
    end

    test "GPU arithmetic overflow recovers through checked native math" do
      if Compute.gpu_detected?() do
        maximum = 3.402_823_466_385_288_6e38
        opts = [gpu: true, gpu_fallback: :error]

        assert {:ok, l2} = Vettore.Distance.normalize([maximum, maximum], :l2, opts)
        assert_in_delta Enum.at(l2, 0), :math.sqrt(0.5), 1.0e-6
        assert_in_delta Enum.at(l2, 1), :math.sqrt(0.5), 1.0e-6

        assert {:ok, minmax} =
                 Vettore.Distance.normalize([-maximum, maximum], :minmax, opts)

        assert_in_delta Enum.at(minmax, 0), 0.0, 1.0e-6
        assert_in_delta Enum.at(minmax, 1), 1.0, 1.0e-6

        assert {:ok, cancelling_dot} =
                 Vettore.Distance.inner_product(
                   [maximum, maximum],
                   [maximum, -maximum],
                   opts
                 )

        assert_in_delta cancelling_dot, 0.0, 1.0e-6

        assert {:error, :metric_overflow} =
                 Vettore.Distance.l2_squared([maximum], [0.0], opts)

        repeated_maximum = f32_binary([maximum, maximum])

        assert {:ok, [pooled_maximum]} =
                 Vector.mean_pool_f32(repeated_maximum, 1, [0, 1],
                   as: :list,
                   gpu: true,
                   gpu_fallback: :error
                 )

        assert pooled_maximum == maximum
      end
    end

    test "GPU scaling preserves tiny vectors, large cosine inputs, and stable z-score" do
      if Compute.gpu_detected?() do
        opts = [gpu: true, gpu_fallback: :error]

        assert {:ok, cosine} = Vettore.Distance.cosine([2.0e19], [1.0], opts)
        assert_in_delta cosine, 1.0, 1.0e-6

        assert {:ok, tiny} = Vettore.Distance.normalize(List.duplicate(1.0e-23, 4), :l2, opts)

        for value <- tiny do
          assert_in_delta value, 0.5, 1.0e-6
        end

        values = [10_000.0, 10_000.1, 9_999.9, 10_000.05]
        assert {:ok, expected} = Vettore.Distance.normalize(values, :zscore, gpu: false)
        assert {:ok, actual} = Vettore.Distance.normalize(values, :zscore, opts)

        for {gpu_value, cpu_value} <- Enum.zip(actual, expected) do
          assert_in_delta gpu_value, cpu_value, 1.0e-5
        end

        left = f32_binary([3.0, 4.0])
        right = f32_binary([6.0, 8.0])
        assert {:ok, 50.0} = Vector.inner_product(left, right, opts)
        assert {:ok, normalized} = Vector.normalize(left, :l2, Keyword.put(opts, :as, :list))
        assert_in_delta Enum.at(normalized, 0), 0.6, 1.0e-5
        assert_in_delta Enum.at(normalized, 1), 0.8, 1.0e-5
      end
    end

    test "compute option errors are tagged before execution" do
      assert {:error, :invalid_gpu_option} =
               Vector.l2([0.0], [1.0], gpu: :sometimes)

      assert {:error, :invalid_gpu_fallback} =
               Vector.normalize([1.0], :l2, gpu_fallback: :sometimes)

      assert {:error, :invalid_gpu_min_size} =
               Vector.mean_pool([[1.0]], gpu: :auto, gpu_min_size: 0)

      assert {:error, :invalid_options} =
               Vector.metric([1.0], [1.0], :l2, gpu: false, gpu: true)

      assert {:error, :invalid_options} = Vector.metric([1.0], [1.0], :l2, :bad)
    end
  end

  defp assert_gpu_parity do
    left = Enum.map(1..20_000, &(&1 / 20_000))
    right = Enum.map(1..20_000, &((20_001 - &1) / 20_000))

    for metric <- [
          :l2,
          :l2_squared,
          :cosine,
          :inner_product,
          :negative_inner_product,
          :manhattan,
          :chebyshev,
          :hamming,
          :jaccard
        ] do
      assert {:ok, cpu} = Vector.metric(left, right, metric, gpu: false)
      assert {:ok, gpu} = Vector.metric(left, right, metric, gpu: true, gpu_fallback: :error)
      assert_in_delta gpu, cpu, max(abs(cpu) * 1.0e-4, 1.0e-5)
    end
  end

  defp f32_binary(values) do
    for value <- values, into: <<>>, do: <<value::float-little-32>>
  end
end
