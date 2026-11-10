import pytest

from bureau_to_workout import bureau, workout


@pytest.fixture(scope="session")
def clean_panel():
    return bureau.simulate_bureau(bureau.SimConfig(n_borrowers=3000, moratorium=False, seed=1))


@pytest.fixture(scope="session")
def moratorium_panel():
    return bureau.simulate_bureau(bureau.SimConfig(n_borrowers=3000, moratorium=True, seed=1))


@pytest.fixture(scope="session")
def workouts():
    return workout.simulate_workouts(workout.WorkoutConfig(n_cases=20_000, seed=3))
