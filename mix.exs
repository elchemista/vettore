defmodule Vettore.MixProject do
  use Mix.Project

  @version "0.3.4"

  def project do
    [
      app: :vettore,
      name: "Vettore",
      version: @version,
      elixir: "~> 1.19",
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      test_coverage: [ignore_modules: [Vettore.Nifs], summary: [threshold: 99]],
      test_ignore_filters: [&String.starts_with?(&1, "test/support/")],
      description: description(),
      package: package(),
      rustler_precompiled: [
        provider: :github,
        owner: "elchemista",
        repo: "vettore",
        tag: "v#{@version}"
      ],
      docs: [
        main: "readme",
        source_ref: "v#{@version}",
        extras: [
          "README.md",
          "CHANGELOG.md",
          "LICENSE"
        ],
        groups_for_modules: [
          "Core API": [Vettore, Vettore.Collection, Vettore.Embedding, Vettore.Result],
          Compatibility: [Vettore.DB],
          "Search and distance": [Vettore.Distance, Vettore.MultiVector],
          Indexes: [Vettore.Index, Vettore.Index.Flat, Vettore.Index.HNSW],
          Storage: [Vettore.Store, Vettore.Store.ETS],
          Encodings: [Vettore.Encoding.Muvera]
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
      {:rustler, "~> 0.38.0", optional: true},
      {:rustler_precompiled, "~> 0.9.0"},
      {:ex_fastembed,
       github: "elchemista/ex_fastembed", branch: "master", only: :test, runtime: false},
      {:credo, "~> 1.7.19", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.5.1", only: :dev},
      # Documentation Provider
      {:ex_doc, "~> 0.40.3", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4.7", only: [:dev, :test], runtime: false}
    ]
  end
end
