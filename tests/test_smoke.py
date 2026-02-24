from parcaestrategy import greet


def test_greet_default():
    assert greet() == "Hello, world!"
