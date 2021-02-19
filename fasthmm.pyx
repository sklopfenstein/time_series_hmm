from cython.parallel cimport prange
from unittest import TestCase
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sklearn.cluster import AffinityPropagation
cimport cython


ln2 = 0.6931471806


class EmissionMatrix:

    def __init__(self, Q0: np.array, y: np.array): # y takes value in the list of all possible obsevrations
        self.coeffs = Q0.astype(np.float128)  # coeffs has dimensions (all possible states X all possible observations)
        self.y = y

    def __call__(self, t, X=None):
        return self.coeffs[X, self.y[t]]

    def __getitem__(self, index):
        return self.coeffs.__getitem__(index)


class TransitionMatrix:

    def __init__(self, P0: np.array):
        self.coeffs = P0.astype(np.float128)

    def __call__(self, X=None, X_prim=None):
        if X is not None and X_prim is not None:
            return self.coeffs[X_prim, X]

    def __getitem__(self, index):
        return self.coeffs.__getitem__(index)

    @property
    def T(self):
        return self.coeffs.T


    
cdef long double[:] log_filtering_recur(long double[:, :] Q, long[:] y, long double[:, :] P,
                                    int t, long double[:] init, long double[:] prev_recur) nogil:
    """Compute the filtering distribution recursively.

    """ 
    cdef:
        int n_states = Q.shape[0]
        long double accu = 0
        int X = 0
        int X_prim = 0
        long double[:] current_step_val = init

    with cython.boundscheck(False):

        for X in range(n_states):

            accu = 0

            for X_prim in range(n_states):
                inter = P[X_prim, X] + prev_recur[X_prim]
                if accu  == 0:
                    accu = inter
                elif accu <= inter:
                   accu = 1 / ln2 * (
                        inter +
                        2**(accu - inter) -
                        (2**(accu - inter))**2 / 2)
                elif accu > inter:
                    accu = 1 / ln2 * (
                        accu +
                        2**(-accu + inter)
                        - (2**(-accu + inter))**2 / 2)
            current_step_val[X] = Q[X, y[t]] + accu
        return current_step_val


cdef long double[:] filtering_recur(long double[:, :] Q, long[:] y, long double[:, :] P,
                                    int t, long double[:] init, long double[:] prev_recur) nogil:
    """Compute the filtering distribution recursively.

    """ 
    cdef:
        int n_states = Q.shape[0]
        long double accu = 0
        int X = 0
        int X_prim = 0
        long double[:] current_step_val = init

    with cython.boundscheck(False):

        for X in range(n_states):

            accu = 0

            for X_prim in range(n_states):
                accu += P[X_prim, X] * prev_recur[X_prim]
            current_step_val[X] = Q[X, y[t]] * accu
        return current_step_val


cdef int compute_log_filtering_distributions(long double[:, :] Q, long[:] y, int T,
                                         long double[:, :] P, long double[:] p0,
                                         long double[:] init, long double[:, :] target,
                                         long double[:] proba_curr_obs_given_past) nogil except -1:
    """Compute the filtering distribution and the scaling coefficient, P(xn | x1 ... xn-1).
    We make hypothesis on the probability to be in each state at t0, and on the emission matrix.
    Then we compute the probability to be in each state at each timestep given the observation
    and the previous probability to be in each state.

    This is the forward phase of the forward backward algorithm.

    params:
       p0 np.ndarray: the initial prior about being in each state.
       Q np.ndarray: the emission matrix (probability of an observation given the state)
       T int: The number of time step
       init np.ndarray: An array of zeros used in the filtering recursion
       target np.ndarray: The proba to be in each state at every time step 
       proba_curr_obs_given_past np.ndarray: The array used to communicate the proba
         for the observation

    """
    cdef:
        int n_states = Q.shape[0]
        int X = 0
        int t = 0

    with cython.boundscheck(False): #ln(a + eps) = ln(a) + (eps) / a - eps2 / 2a2
        # ln(1 + x) = x + 1/2 * x**2 + 1/3 * x**3 ...

        for X in range(n_states):
            # Compute the probability (not normalized) for the joined event of getting
            # the inital observation and being in each state.
            target[0, X] = Q[X, y[0]] + p0[X]
            # Compute the union of all these.
            # Given that the states form a partition of space, we know that this is
            # the probability to observe our first observation.
            # log2(a + b) = (ln(1 + b/a) + ln(a)) / ln2 = 1/ln2 * (ln(a) + (b/a) - 1/2 * (b/a)**2  + 1/3 * (b/a)**3 ...)
            # = 1 / ln2 * (lna + 2**(log2b) / 2**(log2a) - 1/2 * (2**(log2b)/2**(log2a))**2)
            # = 1/ln2 * (lna + 2**(log2b - log2a) - 1/2 * (2**(log2b-log2a))**2)
        if proba_curr_obs_given_past[0] == 0:
            proba_curr_obs_given_past[0] = target[0, X]
        elif proba_curr_obs_given_past[0] <= target[0, X]:
            proba_curr_obs_given_past[0] = 1 / ln2 * (
                target[0, X] +
                2**(proba_curr_obs_given_past[0] - target[0, X]) -
                (2**(proba_curr_obs_given_past[0] - target[0, X]))**2 / 2)
        elif proba_curr_obs_given_past[0] > target[0, X]:
            proba_curr_obs_given_past[0] = 1 / ln2 * (
                proba_curr_obs_given_past[0] +
                2**(-proba_curr_obs_given_past[0] + target[0, X])
                - (2**(-proba_curr_obs_given_past[0] + target[0, X]))**2 / 2)
        # Get the probability to be in each state given the first observation.
        # We have the proba for the joined event, we have the proba for the observation event.
        # Thus we can do the ratio.
        # At this point this number is still big, but be weary of floating point error.
        target[0, :] -= proba_curr_obs_given_past[0]

        # For each time step, do the computation recursively
        for t in range(1, T):
            # Given the new probability to be in each state at t-1 (target[t-1,:]),
            # get the proba for the joined event (observation, state).
            target[t, :] = log_filtering_recur(Q, y, P, t, init, target[t - 1, :])

            for X in range(n_states):
                # get the proba to observe the observation at time step T.
                if proba_curr_obs_given_past[t] == 0:
                    proba_curr_obs_given_past[t] = target[t, X]
                elif proba_curr_obs_given_past[t] <= target[t, X]:
                    proba_curr_obs_given_past[t] = 1 / ln2 * (
                        target[t, X] +
                        2**(proba_curr_obs_given_past[t] - target[t, X]) -
                        (2**(proba_curr_obs_given_past[t] - target[t, X]))**2 / 2)
                elif proba_curr_obs_given_past[t] > target[t, X]:
                    proba_curr_obs_given_past[t] = 1 / ln2 * (
                        proba_curr_obs_given_past[t] +
                        2**(-proba_curr_obs_given_past[t] + target[t, X])
                        - (2**(-proba_curr_obs_given_past[t] + target[t, X]))**2 / 2)
            # Get the probability to be in each state given the Tth observation.
            # This is getting small.
            target[t, :] -= proba_curr_obs_given_past[t]
    return 0

    
cdef int compute_filtering_distributions(long double[:, :] Q, long[:] y, int T,
                                         long double[:, :] P, long double[:] p0,
                                         long double[:] init, long double[:, :] target,
                                         long double[:] proba_curr_obs_given_past) nogil except -1:
    """Compute the filtering distribution and the scaling coefficient, P(xn | x1 ... xn-1).

    params:
       p0 np.ndarray: the initial prior about being in each state.
       Q np.ndarray: the emission matrix (probability of an observation given the state)
       T int: The number of time step
       init np.ndarray: An array of zeros used in the filtering recursion
       target np.ndarray: The proba to be in each state at every time step 
       proba_curr_obs_given_past np.ndarray: The array used to communicate the proba
         for the observation

    """
    cdef:
        int n_states = Q.shape[0]
        int X = 0
        int t = 0

    with cython.boundscheck(False):

        for X in range(n_states):
            # Compute the probability (not normalized) for the joined event of getting
            # the inital observation and being in each state.
            target[0, X] = Q[X, y[0]] * p0[X]
            # Compute the union of all these.
            # Given that the states form a partition of space, we know that this is
            # the probability to observe our first observation.
            proba_curr_obs_given_past[0] += target[0, X]

        # Get the probability to be in each state given the first observation.
        # We have the proba for the joined event, we have the proba for the observation event.
        # Thus we can do the ratio.
        # At this point this number is still big, but be weary of floating point error.
        target[0, :] /= proba_curr_obs_given_past[0]

        # For each time step, do the computation recursively
        for t in range(1, T):
            # Given the new probability to be in each state at t-1 (target[t-1,:]),
            # get the proba for the joined event (observation, state).
            target[t, :] = filtering_recur(Q, y, P, t, init, target[t - 1, :])

            for X in range(n_states):
                # get the proba to observe the observation at time step T.
                proba_curr_obs_given_past[t] += target[t, X]
            # Get the probability to be in each state given the Tth observation.
            # This is getting small.
            target[t, :] /= proba_curr_obs_given_past[t]
    return 0


cdef long double[:] log_smoothing_recur(long double[:, :] P, int t, long double[:, :] filtering,
                                    long double[:] init, long double[:] prev_recur) nogil:
    """Compute the marginal smoothing distribution recursively.

    """
    cdef:
        int n_states = P.shape[0]
        long double future_state_proba_given_current_obs
        long double[:] current_step_proba = init
        int X = 0
        int X_a = 0
        int X_b = 0

    with cython.boundscheck(False):

        for X in range(n_states):
            current_step_proba[X] = 0

            for X_b in range(n_states):

                future_state_proba_given_current_obs = 0

                for X_a in range(n_states):
                    inter = filtering[t, X_a] + P[X_a, X_b]
                    if future_state_proba_given_current_obs  == 0:
                        future_state_proba_given_current_obs = inter
                    elif future_state_proba_given_current_obs <= inter:
                       future_state_proba_given_current_obs = 1 / ln2 * (
                            inter +
                            2**(future_state_proba_given_current_obs - inter) -
                            (2**(future_state_proba_given_current_obs - inter))**2 / 2)
                    elif future_state_proba_given_current_obs > inter:
                        future_state_proba_given_current_obs = 1 / ln2 * (
                            future_state_proba_given_current_obs +
                            2**(-future_state_proba_given_current_obs + inter)
                            - (2**(-future_state_proba_given_current_obs + inter))**2 / 2)

                inter = filtering[t, X] + P[X, X_b] + prev_recur[X_b] - future_state_proba_given_current_obs
                if current_step_proba[X]  == 0:
                    current_step_proba[X] = inter
                elif current_step_proba[X] <= inter:
                    current_step_proba[X] = 1 / ln2 * (
                        inter +
                        2**(current_step_proba[X] - inter) -
                        (2**(current_step_proba[X] - inter))**2 / 2)
                elif current_step_proba[X] > inter:
                    current_step_proba[X] = 1 / ln2 * (
                        current_step_proba[X] +
                        2**(-current_step_proba[X] + inter)
                        - (2**(-current_step_proba[X] + inter))**2 / 2)
                #if future_state_proba_given_current_obs:
                #    current_step_proba[X] /= future_state_proba_given_current_obs
    return current_step_proba


cdef long double[:] smoothing_recur(long double[:, :] P, int t, long double[:, :] filtering,
                                    long double[:] init, long double[:] prev_recur) nogil:
    """Compute the marginal smoothing distribution recursively.

    """
    cdef:
        int n_states = P.shape[0]
        long double future_state_proba_given_current_obs
        long double[:] current_step_proba = init
        int X = 0
        int X_a = 0
        int X_b = 0

    with cython.boundscheck(False):

        for X in range(n_states):
            current_step_proba[X] = 0

            for X_b in range(n_states):

                future_state_proba_given_current_obs = 0

                for X_a in range(n_states):
                    future_state_proba_given_current_obs += (filtering[t, X_a] * P[X_a, X_b])
                current_step_proba[X] += (filtering[t, X] * P[X, X_b] * prev_recur[X_b] /
                                          future_state_proba_given_current_obs)
                #if future_state_proba_given_current_obs:
                #    current_step_proba[X] /= future_state_proba_given_current_obs
    return current_step_proba


cdef int compute_marginal_smoothing_distributions(long double[:, :] P, long double[:, :] filtering,
                                                  long double[:] init, long double[:, :] target) nogil except -1:
    """Compute the marginal smoothing distribution.

    This is the "predictive" step. We make an hypothesis on the transition matrix and use
    this hypothesis to compute P(Next State | Last Observation) for all time steps.

    This is the backward phase of the forward backward algorithm.

    """
    cdef:
        int t = 0
        int X = 0
        int T = filtering.shape[0]
        int n_states = filtering.shape[1]

    with cython.boundscheck(False):

        for X in range(n_states):
            target[T - 1, X] = filtering[T - 1, X]

        for t in range(T - 2, -1, -1):
            target[t, :] = smoothing_recur(P, t, filtering, init, target[t + 1, :])
    return 0


cdef int compute_marginal_log_smoothing_distributions(long double[:, :] P, long double[:, :] filtering,
                                                  long double[:] init, long double[:, :] target) nogil except -1:
    """Compute the marginal smoothing distribution.

    """
    cdef:
        int t = 0
        int X = 0
        int T = filtering.shape[0]
        int n_states = filtering.shape[1]

    with cython.boundscheck(False):

        for X in range(n_states):
            target[T - 1, X] = filtering[T - 1, X]                                         

        for t in range(T - 2, -1, -1):
            target[t, :] = log_smoothing_recur(P, t, filtering, init, target[t + 1, :])
    return 0


class EnhancedState:

    def __init__(self, components):
        self.pattern = Pattern(components)


class Pattern:

    def __init__(self, components):
        self.components = np.c_[components]
        self.median = components.median(axis=0)
        n_components = self.components.shape[0]
        self.percentile_chunks = range(0, 100, 100 // 10)
        self.deviation = np.abs(self.components -
                                np.broadcast_to(self.median,
                                                self.components.shape)).T

    def percentile_from_observation(self, observation):
        T = self.median.shape[0]
        divergence = np.zeros((T,))
        deviation = np.abs(observation - self.median)

        for hour in range(T):
            percentiles = np.zeros(len(self.percentile_chunks))

            for ix, percentile_chunk in enumerate(self.percentile_chunks):
                percentiles[ix] = np.percentile(self.deviation[hour, :], percentile_chunk)

                for low_val, (high, high_val) in zip(percentiles,
                                                 zip(self.percentile_chunks[1:], percentiles[1:])):

                    if deviation[hour] > low_val and deviation[hour] <= high_val:
                    divergence[hour] = high

            if divergence[hour] == 0:
                divergence[hour] = high
        return divergence


def get_hourly_match_in_percentile(pattern, observation):
    T = observation.shape[0]
    hourly_match = pattern.percentile_from_observation(observation)
    return hourly_match


def state_space_representation(enhanced_states, observation):
    T = observation.shape[0]
    n_states = len(enhanced_states)
    hourly_matches = np.zeros((T, n_states))

    for ix, state in enumerate(enhanced_states):
        hourly_matches[:, ix] = get_hourly_match_in_percentile(state.pattern, observation)
    score_by_hour = (100 - hourly_matches)
    match = np.median(score_by_hour, axis=0) / 100
    return match


def make_transition_matrix(classes_df, n_classes):
    counts = classes_df.reset_index().groupby('state').count().sort_index()['date']
    transition_matrix = np.zeros((n_classes, n_classes))

    for cls in counts.index:
        transitions = classes_df[(classes_df == cls).shift(1).fillna(True).values.astype(np.bool).flatten()]\
            .reset_index().groupby('state').count()
        num_tot = transitions.sum()

        for cls2, num in transitions.iterrows():
            transition_matrix[cls, cls2] = num / num_tot
    return TransitionMatrix(transition_matrix)


def make_enhanced_states(observations, classes_df):
    enhanced_states = pd.concat([observations, classes_df], axis=1)\
      .groupby('state')\
      .apply(lambda obs:EnhancedState(obs.drop(columns=['state'])))
    return enhanced_states.values.tolist()


def make_emission_matrix(enhanced_states, observations, discretized, n_states):
    discretized_observations_set = sorted(list({disc.tostring(): disc for disc in discretized}.items()))
    observations_set = {data: (key, obs) for key, (data, obs) in enumerate(discretized_observations_set)}
    y = [observations_set[obs.tostring()][0] for obs in discretized]
    n_obs = len(discretized_observations_set)
    emission_matrix = np.zeros((n_states, n_obs))

    for ixa, state in enumerate(enhanced_states):

        for ixb, (_, _) in enumerate(discretized_observations_set):
            n_components = state.pattern.components.shape[0]
            emission_matrix[ixa, ixb] += n_components
    return EmissionMatrix(emission_matrix / emission_matrix.sum(axis=1)[np.newaxis].T, y)


def unsupervised_state_inference(observations, duck_typed_with_fit_predict):
    hours = range(24)
    trih = np.zeros([observations.shape[0]] * 2)

    for ix, (_, x) in enumerate(observations.sort_index().iterrows()):

         for iy, (_, y) in enumerate(observations.sort_index().iterrows()):

             if ix < iy:
                 trih[ix, iy] = fastdtw(list(zip(hours, x)), list(zip(hours, y)))[0]
    trih_norm = (trih - trih.min()) / (trih.max() - trih.min())
    trih_norm[trih_norm < 0] = 0
    similarities = np.ones(trih_norm.shape[0]) - trih_norm - trih_norm.T
    clusterer = duck_typed_with_fit_predict
    classes_df = pd.Series(clusterer.fit_predict(similarities), name='state', index=observations.sort_index().index)
    counts = pd.Series(classes_df).reset_index().groupby('state').count().sort_index()['date']
    newcounts = counts.sort_index()
    print('counts', newcounts)
    new_class = newcounts.shape[0]

    for ix in counts.index:
        if ix not in newcounts.index:
            classes_df.replace(ix, -1, inplace=True)

    for i, ix in enumerate(newcounts.index):
        classes_df.replace(ix, i, inplace=True)
    classes_df.replace(-1, new_class, inplace=True)
    return classes_df


def initial_parameters(observations, clusterer):
    classes_df = unsupervised_state_inference(observations, clusterer)
    enhanced_states = make_enhanced_states(observations, classes_df)
    discretized = [state_space_representation(enhanced_states, obs) for obs in observations.values]
    discretized_set = [obs[1] for obs in
                       sorted(list({disc.tostring(): disc for disc in discretized}.items()))]
    Q0 = make_emission_matrix(enhanced_states, observations.values, discretized, len(classes_df.unique()))
    P0 = make_transition_matrix(classes_df, len(classes_df.unique()))
    return Q0, P0, discretized, enhanced_states


def viterbi(q, p, y, p0):
    n_states = p.shape[0]
    T = len(y)
    proba = np.zeros((T, n_states))
    z = np.zeros((T,), dtype=np.int32)
    path = np.zeros((T, n_states))

    for X in range(n_states):
        proba[0, X] = p0[X] * q[X, 0]
        path[0, X] = 0

    for t in range(1, T):

        for X in range(n_states):
            calc = proba[t - 1, :] * p[:, X] * q[X, y[t]]
            proba[t, X] = np.max(calc)
            path[t, X] = np.argmax(calc).astype(np.int32)
    z[T - 1] = np.argmax(proba[T - 1, :]).astype(np.int32)

    for t in range(T-2, -1, -1):
        z[t] = path[t, z[t + 1]]
    return z


def logviterbi(q, p, y, p0):
    n_states = p.shape[0]
    T = len(y)
    logproba = np.ones((T, n_states))
    z = np.zeros((T,), dtype=np.int32)
    path = np.zeros((T, n_states))

    for X in range(n_states):
        logproba[0, X] = p0[X] + q[X, 0]
        path[0, X] = 0

    for t in range(1, T):

        for X in range(n_states):
            logcalc = logproba[t - 1, :] + p[:, X] + q[X, y[t]]
            logproba[t, X] = np.max(logcalc)
            path[t, X] = np.argmax(logcalc).astype(np.int32)
    z[T - 1] = np.argmax(logproba[T - 1, :]).astype(np.int32)

    for t in range(T-2, -1, -1):
        z[t] = path[t, z[t + 1]]
    return z, logproba


def viterbi_predict_next_state(Q, P, p0):
    viterbi_path = viterbi(Q.coeffs, P.coeffs, Q.y, p0)
    last_state = viterbi_path[len(Q.y) - 1]
    next_state_proba = P[last_state, :]
    next_state = np.argmax(next_state_proba)
    return next_state, next_state_proba


def viterbi_predict_enhanced_state(Q, P, y, p0, enhanced_states):
    next_state, _ = viterbi_predict_next_state(Q.coeffs, P.coeffs, y, p0)
    return enhanced_states[next_state]


def hard_decision(enhanced_state):
    return enhanced_state.pattern.median


def soft_decision(enhanced_states, probas):
    return np.average(np.c_[[s.pattern.median for s in enhanced_states]], axis=0, weights=probas)


def viterbi_supervised_state_inference(Q, P, p0, observations):
    path = viterbi(Q.coeffs, P.coeffs, Q.y, p0)
    return pd.DataFrame(path, columns=['state'], index=observations.index)


def intermediate_parameters(Q, P, p0, observations):
    classes_df = viterbi_supervised_state_inference(Q, P, p0, observations)
    enhanced_states = make_enhanced_states(observations, classes_df)
    discretized = [state_space_representation(enhanced_states, obs) for obs in observations.values]
    discretized_set = [obs[1] for obs in
                       sorted(list({disc.tostring(): disc for disc in discretized}.items()))]
    Q = make_emission_matrix(enhanced_states, observations.values, discretized, P.coeffs.shape[0])
    P = make_transition_matrix(classes_df, P.coeffs.shape[0])
    return Q, P, discretized, enhanced_states


def viterbi_predict_series(Q, P, p0, observations, enhanced_states, decision='soft'):
    #Q, P, discretized, enhanced_states = intermediate_parameters(Q, P, p0, observations)
    state, state_probas = viterbi_predict_next_state(Q, P, p0)

    if decision == 'soft':
        return soft_decision(enhanced_states, state_probas)

    elif decision == 'hard':
        return hard_decision(enhanced_states[state])


def forward_backward_logprobability_estimates(Q0, P0, y, p0):
    T = y.shape[0]
    n_states = Q0.coeffs.shape[0]
    states = range(n_states)
    Q = Q0
    P = P0
    q_no_null = Q.coeffs
    p_no_null = P.coeffs
    filtering_distributions = np.zeros((T, n_states), dtype=np.float128)
    smoothing_distributions = np.zeros((T, n_states), dtype=np.float128)
    alphas = np.zeros((T, n_states), dtype=np.float128)
    betas = np.zeros((T, n_states), dtype=np.float128)
    gammas = np.zeros((T, n_states), dtype=np.float128)
    ksi_coeffs = np.zeros((T, n_states), dtype=np.float128)
    ksis = np.zeros((T, n_states), dtype=np.float128)
    proba_curr_obs_given_past_obs = np.zeros((T,), dtype=np.float128)
    # Forward phase
    compute_log_filtering_distributions(
        q_no_null, y[:, 0], T, p_no_null, p0.astype(np.float128),
        np.zeros((n_states,), dtype=np.float128),
        filtering_distributions, proba_curr_obs_given_past_obs
    )
    # Backward phase
    compute_marginal_log_smoothing_distributions(
        p_no_null,
        filtering_distributions,
        np.zeros((n_states,), dtype=np.float128),
        smoothing_distributions
    )
    # The gamma coefficient in the HMM standard formalism is the marginal smoothing distribution
    alphas = filtering_distributions  # * P(x1 .. xn)
    gammas = smoothing_distributions
    betas = gammas - alphas  # * P(x1 ... xN)
    ksi_coeffs = betas[1:, :]+ q_no_null[:, y[1:, 0]].T  # t, j
    ksi_coeffs = np.broadcast_to(
        ksi_coeffs.T[np.newaxis].T,
        (T - 1, n_states, n_states)).swapaxes(1, 2)  # t, j -> t, j, i -> t, i, j
    ksi_coeffs = ksi_coeffs +\
      np.broadcast_to(
          alphas[:-1, :].T[np.newaxis].T,
          (T - 1, n_states, n_states)) # t, i -> j, i, t -> t, i, j
    ksi_coeffs = ksi_coeffs +\
      np.broadcast_to(
          p_no_null[np.newaxis],
          (T - 1, n_states, n_states))  # i, j -> t, i, j
    scaling = np.broadcast_to(
        proba_curr_obs_given_past_obs[np.newaxis, np.newaxis].T,
        (T, n_states, n_states))
    ksis = ksi_coeffs - scaling[1:, :, :]
    return np.nan_to_num(gammas), np.nan_to_num(ksis), proba_curr_obs_given_past_obs

    
def forward_backward_probability_estimates(Q0, P0, y, p0):
    T = y.shape[0]
    n_states = Q0.coeffs.shape[0]
    states = range(n_states)
    Q = Q0
    P = P0
    q_no_null = Q.coeffs
    p_no_null = P.coeffs
    filtering_distributions = np.zeros((T, n_states), dtype=np.float128)
    smoothing_distributions = np.zeros((T, n_states), dtype=np.float128)
    alphas = np.zeros((T, n_states), dtype=np.float128)
    betas = np.zeros((T, n_states), dtype=np.float128)
    gammas = np.zeros((T, n_states), dtype=np.float128)
    ksi_coeffs = np.zeros((T, n_states), dtype=np.float128)
    ksis = np.zeros((T, n_states), dtype=np.float128)
    proba_curr_obs_given_past_obs = np.zeros((T,), dtype=np.float128)
    try:
        compute_filtering_distributions(
            q_no_null, y[:, 0], T, p_no_null, p0.astype(np.float128),
            np.zeros((n_states,), dtype=np.float128),
            filtering_distributions, proba_curr_obs_given_past_obs
        )
        compute_marginal_smoothing_distributions(
            p_no_null,
            filtering_distributions,
            np.zeros((n_states,), dtype=np.float128),
            smoothing_distributions
        )
    except ZeroDivisionError:
        raise ZeroDivisionError
    # The gamma coefficient in the HMM standard formalism is the marginal smoothing distribution
    alphas = filtering_distributions  # * P(x1 .. xn)
    gammas = smoothing_distributions
    betas = gammas / alphas  # * P(x1 ... xN)
    ksi_coeffs = betas[1:, :] * q_no_null[:, y[1:, 0]].T  # t, j
    ksi_coeffs = np.broadcast_to(ksi_coeffs.T[np.newaxis].T,
                                 (T - 1, n_states, n_states)).swapaxes(1, 2)  # t, j -> t, j, i -> t, i, j
    ksi_coeffs = ksi_coeffs *\
      np.broadcast_to(alphas[:-1, :].T[np.newaxis].T,
                      (T - 1, n_states, n_states)) # t, i -> j, i, t -> t, i, j
    ksi_coeffs = ksi_coeffs *\
      np.broadcast_to(p_no_null[np.newaxis],
                      (T - 1, n_states, n_states))  # i, j -> t, i, j
    scaling = np.broadcast_to(
        proba_curr_obs_given_past_obs[np.newaxis, np.newaxis].T,
        (T, n_states, n_states))
    ksis = ksi_coeffs / scaling[1:, :, :]
    return np.nan_to_num(gammas), np.nan_to_num(ksis), proba_curr_obs_given_past_obs


def expectation(Q, P, observations, discretized, p0):
    p0_ = p0
    n_states, T = Q.coeffs.shape
    observations_set = {disc.tostring(): disc for disc in discretized}
    observations_set = sorted(list(observations_set.items()))
    observations_set = {data: (key, obs) for key, (data, obs) in enumerate(observations_set)}
    y_disc_observation = np.array([[observations_set[disc.tostring()][0]]
                                   for disc, (_, obs) in zip(discretized, observations.iterrows())])
    gammas, ksis, probas_obs = forward_backward_probability_estimates(Q, P, y_disc_observation, p0_)
    a = ksis.sum(axis=0) / np.broadcast_to(gammas[:-1, :].sum(axis=0)[np.newaxis].T,
                                           (n_states, n_states))  # i -> i, j
    v = np.tile(np.array([key for _, (key, _) in observations_set.items()]), (y_disc_observation.shape[0], 1))
    indicator_function = v == np.tile(y_disc_observation, (1, v.shape[1])).astype(np.int32)
    B = (np.tile(gammas, (indicator_function.shape[1], 1, 1)).swapaxes(0, 1) *\
             np.moveaxis(np.tile(indicator_function, (gammas.shape[1], 1, 1)),
                         (0, 1, 2), (2, 0, 1))).sum(axis=0) / gammas.sum(axis=0)
    p0_ = gammas[0, :]
    return a, B, p0_, probas_obs


def log_expectation(Q, P, observations, discretized, p0):
    p0_ = p0
    n_states, T = Q.coeffs.shape
    observations_set = {disc.tostring(): disc for disc in discretized}
    observations_set = sorted(list(observations_set.items()))
    observations_set = {data: (key, obs) for key, (data, obs) in enumerate(observations_set)}
    y_disc_observation = np.array([[observations_set[disc.tostring()][0]]
                                   for disc, (_, obs) in zip(discretized, observations.iterrows())])
    gammas, ksis, probas_obs = forward_backward_logprobability_estimates(Q, P, y_disc_observation, p0_)
    a = ksis.sum(axis=0) - np.broadcast_to(gammas[:-1, :].sum(axis=0)[np.newaxis].T,
                                           (n_states, n_states))  # i -> i, j
    v = np.tile(np.array([key for _, (key, _) in observations_set.items()]), (y_disc_observation.shape[0], 1))
    indicator_function = v == np.tile(y_disc_observation, (1, v.shape[1])).astype(np.int32)
    B = (np.tile(gammas, (indicator_function.shape[1], 1, 1)).swapaxes(0, 1) +\
             np.moveaxis(np.tile(indicator_function, (gammas.shape[1], 1, 1)),
                         (0, 1, 2), (2, 0, 1))).sum(axis=0) - gammas.sum(axis=0)
    p0_ = gammas[0, :]
    return a, B, p0_, probas_obs


def maximization(a, B, p0, Q, observations):
    Q = EmissionMatrix(B.T, Q.y)
    P = TransitionMatrix(a)
    return Q, P


def baum_welsh(Q0, P0, observations, discretized, p0, eps):
    """EM alogrithm applied to HMMs.

    Description:
    - 

    """
    n_states, n_obs = Q0.coeffs.shape
    Q_prev = EmissionMatrix(np.zeros((n_states, n_obs)), Q0.y)
    P_prev = TransitionMatrix(np.zeros((n_states, n_states)))
    Q, P = Q0, P0
    p0_ = p0
    i = 0
    probas_obs = np.ones((observations.shape[0],))
    probas_obs_prev = np.zeros((observations.shape[0],))
    success = 0
    diverge = 0
    while i <= 1000 and success < 20 and\
        not ((np.abs(probas_obs_prev.sum() - probas_obs.sum()) <= eps) and (probas_obs_prev == 0).all()):
        probas_obs_prev = probas_obs
        try:
            a, B, p0_, probas_obs = expectation(Q, P, observations, discretized, p0_)
        except ZeroDivisionError:
            print('stop condition not reached before float precision error')
            pass
        success = success + 1 if (np.abs(probas_obs_prev.sum() - probas_obs.sum()) <= eps) else 0
        diverge = diverge + 1 if (probas_obs_prev.sum() > probas_obs.sum()) else 0
        Q_prev, P_prev = Q, P
        Q, P = maximization(a, B, p0_, Q, observations)
        i += 1
    return Q, P, p0_


class HiddenMarkovModel:

    def fit(self, y, eps=0.0001, clusterer=None):
        if clusterer is None:
            clusterer =  AffinityPropagation(damping=0.5, max_iter=1000,
                                             convergence_iter=30, copy=True,
                                             preference=None, affinity='precomputed',
                                             verbose=False)
        self.observations = y
        Q0, P0, self.discretized, self.enhanced_states = initial_parameters(y, clusterer)
        p0 = np.array([1 / Q0.coeffs.shape[0]] * Q0.coeffs.shape[0])
        (self.Q, self.P, self.p0
        # self.discretized, self.enhanced_states
        ) = baum_welsh(Q0, P0, y, self.discretized, p0, eps)

    def predict(self, observations):
        return viterbi_predict_series(self.Q, self.P, self.p0, self.observations,
                                      self.enhanced_states, decision='soft')
