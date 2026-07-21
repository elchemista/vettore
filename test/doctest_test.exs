defmodule VettoreDoctestTest do
  use ExUnit.Case, async: true

  doctest Vettore
  doctest Vettore.Distance
  doctest Vettore.MultiVector
end
