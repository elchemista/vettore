defmodule VettoreDoctestTest do
  use ExUnit.Case, async: true

  doctest Vettore
  doctest Vettore.Compute
  doctest Vettore.Distance
  doctest Vettore.MultiVector
  doctest Vettore.Vector
end
