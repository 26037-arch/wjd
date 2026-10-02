import numpy as np

from conical.test_models import box, cylinder, generate_random_model


def test_deterministic_by_index_and_seed():
    a, ma = generate_random_model(20261002, 17)
    b, mb = generate_random_model(20261002, 17)
    assert ma == mb
    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.faces, b.faces)


def test_known_answer_solids():
    cuboid = box(4, 5, 6)
    round_prism = cylinder(3, 8)
    assert cuboid.is_watertight and abs(cuboid.volume - 120) < 1e-8
    assert round_prism.is_watertight
    assert abs(round_prism.volume - np.pi * 3**2 * 8) < 3
