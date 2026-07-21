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
      {:rustler, "~> 0.36.1", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      {:ex_fastembed, github: "elchemista/ex_fastembed", branch: "master", only: :test},
      # Documentation Provider
      {:ex_doc, "~> 0.40.3", only: :dev, optional: true, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end
end
