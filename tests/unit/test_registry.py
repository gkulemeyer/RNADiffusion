from src.core.registry import Registry


def test_registry_register_and_get():
    reg = Registry("Test")

    @reg.register("foo")
    class Foo:
        pass

    assert "foo" in reg
    assert reg.get("foo") is Foo
