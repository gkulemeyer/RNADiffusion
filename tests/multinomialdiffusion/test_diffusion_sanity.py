import torch as tr

from src.core.diffusion import MultinomialDiffusionModel


class _DummyModel(tr.nn.Module):
    def forward(self, x, t):
        b, _, h, w = x.shape
        return tr.zeros((b, 2, h, w), device=x.device)


def _build_model(timesteps=3):
    return MultinomialDiffusionModel(
        num_classes=2,
        time_steps=timesteps,
        schedule="cosine",
        model=lambda **kwargs: _DummyModel(),
    )


def test_q_pred_and_q_step_normalized():
    model = _build_model()
    x0 = tr.zeros((2, 2, 4, 4))
    t = tr.zeros((2,), dtype=tr.long)
    q = model.q_pred(x0, t)
    assert tr.all(q >= 0)

    xt = tr.zeros((2, 4, 4), dtype=tr.long)
    q2 = model.q_step(xt, t)
    s = q2.sum(dim=1)
    assert tr.allclose(s, tr.ones_like(s), atol=1e-5)


def test_q_posterior_normalized():
    model = _build_model()
    x0 = tr.zeros((2, 4, 4), dtype=tr.long)
    xt = tr.zeros((2, 4, 4), dtype=tr.long)
    t = tr.zeros((2,), dtype=tr.long)
    post = model.q_posterior(x0, xt, t)
    s = post.sum(dim=1)
    assert tr.allclose(s, tr.ones_like(s), atol=1e-5)


def test_sample_shape():
    model = _build_model()
    cond = tr.zeros((2, 16, 4, 4))
    out = model._sample(cond)
    assert out.shape == (2, 4, 4)


def test_compute_vlb_masked():
    model = _build_model()
    x0 = tr.zeros((2, 2, 4, 4))
    xt = tr.zeros((2, 4, 4), dtype=tr.long)
    t = tr.zeros((2,), dtype=tr.long)
    cond = tr.zeros((2, 16, 4, 4))
    mask = tr.zeros((2, 1, 4, 4))
    loss = model.compute_vlb(x0, xt, t, cond, mask=mask)
    assert tr.isfinite(loss)
