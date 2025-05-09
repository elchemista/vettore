defmodule Vettore.MixProject do
  use Mix.Project

  @version "0.2.1"

  def project do
    [
      app: :vettore,
      name: "Vettore",
      version: @version,
      elixir: "~> 1.18",
      build_embedded: Mix.env() == :prod,
      start_permanent: Mix.env() == :prod,
      deps: deps(),
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
      extra_applications: []
    ]
  end

  defp description() do
    "Vettore: In-Memory Vector Database with Elixir & Rustler"
  end

  defp package() do
    [
      name: "vettore",
      maintainers: ["Yuriy Zhar"],
      files: ~w(
             lib
             mix.exs
             README.md
             LICENSE
             checksum-Elixir.Vettore.Nifs.exs
             native/vettore/Cargo.toml
             native/vettore/src
             priv/native/*.so
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
      # {:rustler, "~> 0.36.1"},
      {:rustler, ">= 0.0.0", optional: true},
      {:rustler_precompiled, "~> 0.8"},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      # Documentation Provider
      {:ex_doc, "~> 0.28.3", only: [:dev, :test], optional: true, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end
end
