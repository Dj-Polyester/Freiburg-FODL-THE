import numpy as np
from lib.mcmc import *


def mcmc_test():
    np.random.seed(11111)
    start_sample = 0.5
    num_samples = 20
    burn_in = 100
    sampling_freq = 50
    sample_list = metropolis_hastings(target_2gmm1d_pdf_upto_norm, proposal_gaussian1d_pdf,
                                      proposal_gaussian1d_sampler, start_sample, num_samples,
                                      burn_in, sampling_freq)
    assert len(sample_list) == num_samples
    true_sample_list = np.array([-3.2689687771162386, -3.180743757717261, -4.756687936060402, -4.78725429777373,
                                 -6.7485546257704225, -4.706753270320401, -0.9954377871617295, -0.2521156604643204,
                                 4.008900495010325, 8.951910625376797, 9.535445525739686, 7.848019149851967,
                                 1.8904364301524361, 4.281240643773221, 3.894560117768664, 4.547595794516745,
                                 7.623259014820943, 6.301494663089001, 5.301780576848071, 0.8300030947108685])
    err_msg = "Incorrect metropolis_hastings implementation, please check if you have followed the code comments"
    np.testing.assert_allclose(true_sample_list, sample_list, rtol=1e-7, err_msg=err_msg)


if __name__ == "__main__":

    mcmc_test()
    print('Test complete.')
