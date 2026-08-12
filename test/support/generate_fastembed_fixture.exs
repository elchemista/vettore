Code.require_file("fastembed_fixture.ex", __DIR__)

path = Vettore.Test.FastembedFixture.write!()
Mix.shell().info("Generated #{path}")
