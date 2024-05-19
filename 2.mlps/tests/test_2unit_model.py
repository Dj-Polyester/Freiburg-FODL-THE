from lib.models import create_2unit_net, run_test_model
from lib.network import Sequential


def test_2unit_model():
    """Create the 2 unit 2 layer network and test it"""
    model = create_2unit_net()
    assert isinstance(model, Sequential), f"model should be Sequential but is {type(model)}"
    run_test_model(model)


if __name__ == '__main__':
    test_2unit_model()
    print("Test complete.")
