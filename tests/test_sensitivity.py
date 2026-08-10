import pytest
from src.actuarial_risk_model.sensitivity import one_way_sensitivity, two_way_sensitivity


def test_one_way_sensitivity() -> None:
    def compute(params):
        return params['frequency'] * params['severity']

    result = one_way_sensitivity(
        base_params={'frequency': 2, 'severity': 100},
        param_name='frequency',
        values=[1, 2, 3],
        compute_fn=compute,
    )
    assert result['base_output'] == 200
    assert [r['output'] for r in result['results']] == [100, 200, 300]


def test_two_way_sensitivity_grid_shape() -> None:
    def compute(params):
        return params['x'] + params['y']

    result = two_way_sensitivity(
        base_params={'x': 0, 'y': 0},
        param_x='x', values_x=[1, 2],
        param_y='y', values_y=[10, 20, 30],
        compute_fn=compute,
    )
    assert len(result['grid']) == 2
    assert len(result['grid'][0]) == 3
    assert result['grid'][0][0] == 11
    assert result['grid'][1][2] == 32
