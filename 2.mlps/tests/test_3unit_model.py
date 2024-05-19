from lib.models import create_3unit_net, run_test_model
from lib.network import Sequential


def test_3unit_model():
    """Create the 3 unit 2 layer network and test it"""
    model = create_3unit_net()
    assert isinstance(model, Sequential), f"model should be Sequential but is {type(model)}"
    run_test_model(model)
    assert len(model.modules) == 3, f"model should have 2 layers but has {len(model.modules)} layers"


if __name__ == '__main__':
    test_3unit_model()
    print("Test complete.")
