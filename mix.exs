defmodule Vettore.MixProject do
  use Mix.Project

  def project do
    [
      app: :vettore,
      name: "Vettore - In-Memory Vector Database",
      version: "0.1.5",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: [
        main: "Vettore",
        extras: [
          "README.md",
          "LICENSE"
        ]
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp description() do
    "Vettore: In-Memory Vector Database with Elixir & Rustler"
  end

  defp package() do
    [
      licenses: ["MIT"],
      links: %{
        project: "https://github.com/elchemista/vettore",
        developer_github: "https://github.com/elchemista"
      }
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.36.1"},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false},
      {:benchee, "~> 1.0", only: :dev},
      # Documentation Provider
      {:ex_doc, "~> 0.28.3", only: [:dev, :test], optional: true, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end
end
