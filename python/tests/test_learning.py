import torch

from gpu2tensor.examples.pretrain import normalize_training_features


def test_constant_training_counter_does_not_amplify_unseen_values():
    features = torch.tensor([[2.0, 1.0], [2.0, 3.0], [999.0, 1000.0]])
    train = torch.tensor([True, True, False])
    normalized, mean, scale, active = normalize_training_features(features, train)
    assert active.tolist() == [False, True]
    assert normalized[:, 0].tolist() == [0.0, 0.0, 0.0]
    torch.testing.assert_close(mean, torch.tensor([2.0, 2.0]))
    assert torch.isfinite(normalized).all()
    altered = features.clone()
    altered[2] = -10000
    _, other_mean, other_scale, other_active = normalize_training_features(altered, train)
    torch.testing.assert_close(mean, other_mean)
    torch.testing.assert_close(scale, other_scale)
    torch.testing.assert_close(active, other_active)
