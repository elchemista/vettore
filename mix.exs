defmodule Vettore.MixProject do
  use Mix.Project

  @version "0.3.2"

  def project do
    [
      app: :vettore,
      name: "Vettore",
      version: @version,
      elixir: "~> 1.19",
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      test_coverage: [ignore_modules: [Vettore.Nifs], summary: [threshold: 98]],
      description: description(),
      package: package(),
      rustler_precompiled: [
        provider: :github,
        owner: "elchemista",
        repo: "vettore",
        tag: "v#{@version}"
      ],
      docs: [
        master: "readme",
        extras: [
          "README.md",
          "CHANGELOG.md",
          "LICENSE"
        ]
      ],
      source_url: "https://github.com/elchemista/vettore",
      homepage_url: "https://github.com/elchemista/vettore"
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      mod: {Vettore.Application, []},
      extra_applications: []
    ]
  end

  defp description() do
    "Adaptive vector search for Elixir with ETS storage and native acceleration"
  end

  defp package() do
    [
      name: "vettore",
      maintainers: ["Yuriy Zhar"],
      files: ~w(
             lib
             mix.exs
             README.md
             CHANGELOG.md
             RELEASE.md
             LICENSE
             checksum-*.exs
             native/vettore/Cargo.toml
             native/vettore/Cargo.lock
             native/vettore/src
      ),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/elchemista/vettore"
      }
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      rustler_dependency(),
      {:rustler_precompiled, "~> 0.9.0"},
      {:credo, "~> 1.7.19", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.5.1", only: :dev},
      # Documentation Provider
      {:ex_doc, "~> 0.40.3", only: :dev, optional: true, runtime: false},
      {:dialyxir, "~> 1.4.7", only: [:dev, :test], runtime: false}
    ] ++ fastembed_test_dependency()
  end

  # ex_fastembed still constrains its test-only Mix dependency to Rustler 0.36.
  # Keep that compatibility override out of Vettore's published Hex metadata.
  defp rustler_dependency do
    if fastembed_test?() do
      {:rustler, "~> 0.38.0", optional: true, override: true}
    else
      {:rustler, "~> 0.38.0", optional: true}
    end
  end

  defp fastembed_test_dependency do
    if fastembed_test?() do
      [
        {:ex_fastembed,
         github: "elchemista/ex_fastembed", branch: "master", only: :test, runtime: false}
      ]
    else
      []
    end
  end

  defp fastembed_test?, do: System.get_env("VETTORE_TEST_EX_FASTEMBED") == "1"
end
