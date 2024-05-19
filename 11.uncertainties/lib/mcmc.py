import numpy as np
from typing import Callable, Optional, List
import scipy
from scipy.stats import norm


def accept_with_prob(prob: float) -> bool:
    """
    returns True/False denoting whether to accept/reject a point x respectively, given that the probability of
    acceptance is prob

    Args:
        prob: probability of accepting any point x

    Returns:
        True if x is accepted, else False

    """
    return np.random.uniform() < prob


def metropolis_hastings(
    target_pdf: Callable[[float], float],
    proposal_pdf: Callable[[float, float], float],
    proposal_sampler: Callable[[float], float],
    start_sample: float,
    num_samples: int,
    burn_in: int = 50,
    sampling_freq: int = 10,
) -> List[float]:
    """
    Here we implement the Metropolis hastings mcmc algorithm that can sample from any probability
    distribution that we can evaluate up to a constant.

    Args:
        target_pdf: A function that takes a point x as input and outputs the probability of x up to a constant.
        We want to sample points from this target distribution
        proposal_pdf: A function that takes points x1 and x2 as input and outputs g(x2|x1) from the slides
        proposal_sampler: A function that takes a point x1 as input and samples a point x2 from the distribution
        g(x|x1)
        start_sample: The initital sample to start the mcmc sampling from. This is same as theta_0 from the slides
        num_samples: Number of samples to generate from the target_pdf
        burn_in: number of samples determening the burning in phase
        sampling_freq: The frequency t for periodic resampling

    Returns:
        The sampled_points list containing point sampled according to target_pdf
    """
    # Initially we only have the start_sample
    epsilon = 1e-12
    sampled_points = [start_sample]
    current_sample = start_sample
    n_samples_to_generate = burn_in + (num_samples - 1) * sampling_freq + 1
    for i in range(n_samples_to_generate):
        # START TODO ##########
        # sample a candidate point from proposal_sampler given the current_sample
        # Accept the candidate with probability given by the acceptance ratio. Add epsilon to denominator of
        # acceptance ration for numerical stability
        # To accept a candidate with certain probability, use the function accept_with_prob()
        # Add the generated samples to the list sampled_points

        # Test file keeps giving AssertionError saying that I have 18 samples,
        # whereas the correct number of samples should be 20. I assume
        # this is due to either a mistake I overlooked, or the ranges
        # were set differently (1 more or less maybe)?
        def mh_single_iter():
            nonlocal current_sample
            current_candidate_sample = proposal_sampler(current_sample)
            # eval_func(t1,t2) is p_target(t1)*g(t2|t1)
            eval_func = lambda t1, t2: target_pdf(t1) * proposal_pdf(t1, t2)
            candidate_eval = eval_func(current_candidate_sample, current_sample)
            current_eval = eval_func(current_sample, current_candidate_sample)
            acceptance_ratio = candidate_eval / (current_eval + epsilon)
            acceptance_prob = min(1, acceptance_ratio)
            accepted = accept_with_prob(acceptance_prob)
            # Only add if accepted
            if accepted:
                sampled_points.append(current_candidate_sample)
                current_sample = current_candidate_sample

        mh_single_iter()
    # END TODO ##########
    sampled_points = burn_in_periodic_sampling(
        sampled_points, k=burn_in, t=sampling_freq
    )
    return sampled_points


def burn_in_periodic_sampling(
    mcmc_sampled_points: List[float], k: float = 50, t: float = 10
) -> List[float]:
    """
    Implement periodic sampling with burning in phase, for the samples obtained from mcmc
    Args:
        mcmc_sampled_points: points originally sampled from mcmc approach
        k: burn in phase time (in number of samples)
        t: frequency for periodic sampling

    Returns:
        Points from target distribution that are now sampled more independently
    """
    sampled_points = []
    i = k
    while i < len(mcmc_sampled_points):
        sampled_points.append(mcmc_sampled_points[i])
        i += t
    return sampled_points


def target_3gmm1d_pdf_upto_norm(x: float) -> float:
    """
    A 1d gaussian mixture model which is our target distribution computable up to a normalisation constant
    3 gaussians with same std-dev = 2.0 and means -4.0, 12.0, 6.0 with weights 0.25, 0.25, 0.5 respectively
    """
    z1 = (x + 4.0) / 2.0
    z2 = (x - 12.0) / 2.0
    z3 = (x - 6.0) / 2.0
    return (
        (0.25 * np.exp(-0.5 * z1 * z1))
        + (0.25 * np.exp(-0.5 * z2 * z2))
        + (0.5 * np.exp(-0.5 * z3 * z3))
    )


def target_2gmm1d_pdf_upto_norm(x: float) -> float:
    """
    A 1d gaussian mixture model which is our target distribution computable up to a normalisation constant
    2 gaussians with same std-dev = 2.0 and means -5.0, 5.0 with weights 0.5, 0.5 respectively
    """
    z1 = (x + 5.0) / 2.0
    z2 = (x - 5.0) / 2.0
    return (0.5 * np.exp(-0.5 * z1 * z1)) + (0.5 * np.exp(-0.5 * z2 * z2))


def proposal_gaussian1d_pdf(x1: float, x2: float) -> float:
    """
    A gaussian pdf
    """
    return norm.pdf(x2, loc=x1, scale=0.5)


def proposal_gaussian1d_sampler(x1: float) -> float:
    """
    return a sample from proposal gaussian pdf
    """
    return np.random.normal(loc=x1, scale=0.5)


def target_2gmm2d_pdf_upto_norm(x: List[float]) -> float:
    """
    A 2d gaussian mixture model which is our target distribution computable up to a normalisation constant
    2 gaussians with variances 1.0, and means [-3.0, -3.0], [3.0, 3.0] with weights 0.5, 0.5 respectively
    """
    mu1 = [-2.0, -2.0]
    mu2 = [2.0, 2.0]
    z1 = (x[0] - mu1[0]) / 1.0
    z2 = (x[1] - mu1[1]) / 1.0
    p1 = np.exp(-0.5 * ((z1 * z1) + (z2 * z2)))
    z1 = (x[0] - mu2[0]) / 1.0
    z2 = (x[1] - mu2[1]) / 1.0
    p2 = np.exp(-0.5 * ((z1 * z1) + (z2 * z2)))
    return p1 + p2


def proposal_gaussian2d_pdf(x1: List[float], x2: List[float]) -> float:
    """
    A gaussian pdf
    """
    return norm.pdf(x2[0], loc=x1[0], scale=0.5) * norm.pdf(x2[1], loc=x1[1], scale=0.5)


def proposal_gaussian2d_sampler(x1: List[float]) -> List[float]:
    """
    return a sample from proposal gaussian pdf
    """
    a = np.random.normal(loc=x1[0], scale=0.5)
    b = np.random.normal(loc=x1[1], scale=0.5)
    return [a, b]
