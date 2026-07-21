defmodule Vettore.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {DynamicSupervisor, strategy: :one_for_one, name: Vettore.ETSSupervisor}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Vettore.Supervisor)
  end
end
