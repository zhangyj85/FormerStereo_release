from .solver_train import TrainSolver
from .solver_test import TestSolver
from .solver_submit import SubmitSolver


def get_solver(config):

    mode_cfg = config['mode'].lower()
    if mode_cfg == 'train':
        solver = TrainSolver(config)
    elif mode_cfg == 'test':
        solver = TestSolver(config)
    elif mode_cfg == 'submit':
        solver = SubmitSolver(config)
    else:
        raise NotImplementedError('Solver [{:s}] is not supported.'.format(mode_cfg))
    
    return solver
