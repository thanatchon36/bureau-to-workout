"""Bureau simulator and scoring metrics."""
import numpy as np

from bureau_to_workout import bureau


def test_same_seed_same_panel(clean_panel):
    again = bureau.simulate_bureau(bureau.SimConfig(n_borrowers=3000, moratorium=False, seed=1))
    np.testing.assert_array_equal(again.s_obs, clean_panel.s_obs)
    np.testing.assert_array_equal(again.applications.to_numpy(), clean_panel.applications.to_numpy())


def test_without_moratorium_reports_the_truth(clean_panel):
    assert clean_panel.s_obs.shape == (3000, clean_panel.cfg.n_months)
    assert not clean_panel.mor.any()
    np.testing.assert_array_equal(clean_panel.s_obs, clean_panel.s_true)


def test_moratorium_masks_arrears_inside_the_shock(moratorium_panel):
    p = moratorium_panel
    months = np.flatnonzero(p.mor.any(axis=0))
    assert months.min() >= p.cfg.shock_start and months.max() <= p.cfg.shock_end
    # deferred borrowers are reported as less delinquent than they are
    assert p.s_obs[p.mor].mean() < p.s_true[p.mor].mean()


def test_feature_cube_matches_panel(clean_panel):
    cube = bureau.build_feature_cube(clean_panel)
    for name, arr in cube.items():
        assert arr.shape[0] == 3000, name


def test_metrics_on_known_cases():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert bureau.psi(x, x) < 1e-6
    assert bureau.psi(x, x + 1.0) > 0.25
    y = np.r_[np.zeros(500), np.ones(500)]
    assert bureau.ks_stat(y, y) == 1.0
    p = rng.uniform(size=200_000)
    assert bureau.ece((rng.uniform(size=p.size) < p).astype(float), p) < 0.01


def test_scorecard_json_round_trip(clean_panel):
    """The scorecard ships as plain JSON on Hugging Face; reloading it must score identically."""
    import json

    cube = bureau.build_feature_cube(clean_panel)
    dev = bureau.snapshot_dataset(clean_panel, cube, [12, 15, 18, 21, 24])
    feats = [f for f in bureau.FEATURES if dev[f].std() > 0]
    sc = bureau.WoEScorecard().fit(dev[feats], dev["y_obs"])
    again = bureau.WoEScorecard.from_dict(json.loads(json.dumps(sc.to_dict())))
    np.testing.assert_allclose(again.predict_proba(dev[feats]), sc.predict_proba(dev[feats]))
    np.testing.assert_allclose(again.score(dev[feats]), sc.score(dev[feats]))
    assert again.reason_codes(dev[feats].head(5)) == sc.reason_codes(dev[feats].head(5))
