defmodule Vettore.MixProject do
  use Mix.Project

  def project do
    [
      app: :vettore,
      version: "0.1.1",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:rustler, "~> 0.36.1"},
      {:credo, "~> 1.6", only: [:dev, :test], runtime: false}
    ]
  end
end
