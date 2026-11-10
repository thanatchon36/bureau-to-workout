"""Workout simulator, budgeted policies and off-policy evaluation."""
import numpy as np

from bureau_to_workout import workout


def test_logged_propensities_are_probabilities(workouts):
    np.testing.assert_allclose(workouts.e_true.sum(axis=1), 1.0)
    assert workouts.e_true.min() > 0  # overlap holds by construction
    assert set(np.unique(workouts.A)) == {0, 1, 2, 3}


def test_budget_policy_never_overspends(workouts):
    rng = np.random.default_rng(0)
    tau = rng.normal(0, 0.05, size=(len(workouts.A), 4))
    tau[:, 0] = 0
    for pct in (0, 2, 10):
        budget = pct / 100 * workouts.balance.sum()
        policy = workout.budget_policy(tau, workouts.balance, budget)
        assert workout.spend(policy, workouts.balance) <= budget + 1e-6
    assert set(np.unique(workout.budget_policy(tau, workouts.balance, 0.0))) <= {0, 1}


def test_budget_policy_writes_down_where_it_pays_most():
    balance = np.full(4, 100.0)
    tau = np.array([[0, 0.0, 0.10, 0.0], [0, 0.0, 0.30, 0.0], [0, 0.05, 0.0, 0.0], [0, 0.0, -0.1, -0.1]])
    policy = workout.budget_policy(tau, balance, budget=30.0)  # one write-down of 30% of 100
    assert policy.tolist() == [0, 2, 1, 0]


def test_doubly_robust_ope_recovers_true_value(workouts):
    idx = np.arange(len(workouts.A))
    for arm in (0, 1):
        policy = np.full(len(workouts.A), arm)
        truth = workout.true_value(workouts, idx, policy)["recovery (% of balance)"]
        _, ips, dr = workout.ope(policy, workouts.ev, workouts.e_true, workouts.A, workouts.Y, workouts.balance)
        assert abs(dr - truth) < 0.5
        assert abs(ips - truth) < 2.0
