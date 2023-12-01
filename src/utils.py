import numpy as np

from scipy.optimize import curve_fit
from scipy.optimize import minimize

from functools import partial


def nk_isocratic_model(
    params: tuple[float, float, float], 
    phi: float|np.ndarray
) -> float|np.ndarray:

    '''Neue-kuss isocratic retention model.

    Args:
        params:
            The Neue-Kuss parameters (S1, S2, kw).
        phi:
            The fraction of strong eluent.

    Returns:
        Retention factors.
    '''

    S1, S2, kw = params

    return kw * (1 + S2 * phi)**2 * np.exp(-(S1 * phi) / (1 + S2 * phi))

def lss_isocratic_model(
    params: tuple[float, float], 
    phi: float|np.ndarray
) -> float|np.ndarray:

    '''LSS isocratic retention model.

    Args:
        params:
            The LSS parameters (S, kw).
        phi:
            The fraction of strong eluent.

    Returns:
        Retention factors.
    '''

    S, kw = params
    return kw * np.exp(- S * phi)

def nk_gradient_model(
    params: tuple[float, float, float], 
    phi_start: float|np.ndarray, 
    phi_end: float|np.ndarray, 
    gradient_duration: float|np.ndarray,
    dwell_time: float = 0.614, 
    void_time: float = 0.258
) -> tuple[float|np.ndarray, float|np.ndarray]:

    '''Neue-Kuss gradient retention model.
     
    Has three "phases": `k` before the gradient, `k` during the gradient, 
    and `k` after the gradient.

    Args:
        params:
            The Neue-Kuss parameters (S1, S2, kw).
        phi_start:
            The fraction of strong eluent at the start of the gradient.
        phi_end:
            The fraction of strong eluent at the end of the gradient.
        gradient_duration:
            The duration (min) of the gradient (from `phi_start` to `phi_end`).
        dwell_time:
            The dwell time of the system (time delay for the gradient). 
        void_time:
            The void time of the system (the time it takes for an unretained
            molecule to make it from the injection to the detector). 

    Returns:
        Retention factors and phi at elution.
    '''

    epsilon = 1e-7

    S1, S2, kw = params

    beta = (phi_end - phi_start) / gradient_duration

    k_start = nk_isocratic_model((S1, S2, kw), phi_start)
    k_end = nk_isocratic_model((S1, S2, kw), phi_end)

    # Elution before start of gradient
    k_before_gradient = k_start

    # Elution during the gradient
    phi_elution_numerator = (
        phi_start + (1 + S2 * phi_start) / S1 * np.log(
            1 + S1 * beta * kw *
            np.exp(-(S1 * phi_start) / (1 + S2 * phi_start)) *
            (void_time - dwell_time / k_start)
        )
    )
    phi_elution_denominator = (
        1 - (S2 * (1 + S2 * phi_start)) / S1 * np.log(
            1 + S1 * beta * kw *
            np.exp(-(S1 * phi_start) / (1 + S2 * phi_start)) *
            (void_time - dwell_time / k_start)
        )
    )
    phi_elution = phi_elution_numerator / phi_elution_denominator

    k_during_gradient = (
        dwell_time/void_time + (phi_elution - phi_start) / (beta*void_time))

    # Elution after the gradient
    k_after_gradient = (
        k_end - k_end / k_start * dwell_time / void_time +
        dwell_time / void_time +
        gradient_duration / void_time -
        k_end / (beta * void_time * kw * S1) * (
            np.exp((S1 * phi_end) / (1 + S2 * phi_end)) -
            np.exp((S1 * phi_start) / (1 + S2 * phi_start))
        )
    )

    condition_1 = np.logical_or(
        np.logical_and(
            (void_time) < (k_start * void_time + void_time),
            (k_start * void_time + void_time) < (void_time + dwell_time)),
        np.logical_or(
            (np.abs(phi_end - phi_start) < epsilon),
            (gradient_duration < epsilon)))

    condition_2 = np.logical_and(
        (void_time + dwell_time) < (k_start * void_time + void_time),
        (k_during_gradient * void_time + void_time) < (void_time + dwell_time + gradient_duration))

    k = np.where(
        condition_1,
        k_before_gradient,
        np.where(
            condition_2,
            k_during_gradient,
            k_after_gradient
        )
    )

    tr = k * void_time + void_time # retention time
    phi_at_elution = np.where(
        tr <= (void_time + dwell_time),
        phi_start,
        np.where(
            tr > (void_time + dwell_time + gradient_duration),
            phi_end,
            phi_elution
        )
    )
    return k, phi_at_elution

def lss_gradient_model(
    params: tuple[float, float, float], 
    phi_start: float|np.ndarray, 
    phi_end: float|np.ndarray, 
    gradient_duration: float|np.ndarray,
    dwell_time: float = 0.614, 
    void_time: float = 0.258
) -> tuple[float|np.ndarray, float|np.ndarray]:

    '''LSS gradient retention model.

    Has three "phases": `k` before the gradient, `k` during the gradient, 
    and `k` after the gradient.

    Args:
        params:
            The LSS parameters (S, kw).
        phi_start:
            The fraction of strong eluent at the start of the gradient.
        phi_end:
            The fraction of strong eluent at the end of the gradient.
        gradient_duration:
            The duration (min) of the gradient (from `phi_start` to `phi_end`).
        dwell_time:
            The dwell time of the system (time delay for the gradient). 
        void_time:
            The void time of the system (the time it takes for an unretained
            molecule to make it from the injection to the detector). 

    Returns:
        Retention factors and phi at elution (currently `None`).
    '''

    epsilon = 1e-7

    S, kw = params

    k_start = lss_isocratic_model((S, kw), phi_start)
    k_end = lss_isocratic_model((S, kw), phi_end)

    # beta = slope of gradient
    beta = (phi_end - phi_start) / gradient_duration
    b = S * beta * void_time

    k_before_gradient = k_start
    k_during_gradient = dwell_time / void_time + 1 / b * np.log((void_time - dwell_time / k_start) * b / void_time * k_start + 1)
    k_after_gradient = k_end - k_end / k_start * dwell_time / void_time + (dwell_time + gradient_duration) / void_time + k_end / k_start * 1 / b * (1 - np.exp(b * gradient_duration / void_time))

    condition_1 = np.logical_or(
        np.logical_and(
            (void_time) < (k_start * void_time + void_time),
            (k_start * void_time + void_time) < (void_time + dwell_time)),
        np.logical_or(
            (np.abs(phi_end - phi_start) < epsilon),
            (gradient_duration < epsilon)))

    condition_2 = np.logical_and(
        (void_time + dwell_time) < (k_start * void_time + void_time),
        (k_during_gradient * void_time + void_time) < (void_time + dwell_time + gradient_duration))

    k = np.where(
        condition_1,
        k_before_gradient,
        np.where(
            condition_2,
            k_during_gradient,
            k_after_gradient
        )
    )

    return k, None

def fit_retention_model(
    xdata: tuple[np.ndarray, ...], 
    ydata: tuple[float, ...], 
    dwell_time: float = 0.614, 
    void_time: float = 0.258
) -> tuple[np.ndarray, ...]:

    '''Fits a NK gradient model to the data.'''

    def lss_model(x, S, kw, divide_by):
        phi_start, phi_end, gradient_duration = x
        return lss_gradient_model(
            [S, kw], phi_start, phi_end, gradient_duration,
            dwell_time, void_time)[0] / divide_by

    def nk_model(x, S1, S2, kw, divide_by):
        phi_start, phi_end, gradient_duration = x
        return nk_gradient_model(
            [S1, S2, kw], phi_start, phi_end, gradient_duration,
            dwell_time, void_time)[0] / divide_by

    initial_param = [10., 1., 100.]

    try:
        # Perform curve-fitting
        # First obtain good initial parameter values based on a lss model fit
        initial_param = curve_fit(
            f=partial(lss_model, divide_by=ydata),
            xdata=xdata,
            ydata=ydata / ydata,
            maxfev=2000,
            p0=[1., 1.],
            bounds=(0., np.inf))[0]

        initial_param = list(initial_param)
        initial_param.insert(1, 1.0) # add S2 to initial params

        # Then fit the NK model, initialized from these values
        nk_fitted_param = curve_fit(
            f=partial(nk_model, divide_by=ydata),
            xdata=xdata,
            ydata=ydata / ydata,
            maxfev=2000,
            p0=initial_param,
            bounds=(0., np.inf))[0]

        # Calculate the mean relative error between true and predicted values
        error = np.mean(
            np.abs(ydata - nk_model(xdata, *nk_fitted_param, divide_by=1.)) / ydata)

        # Perform Nelder-Mead optimization instead if error is too high
        if error > 0.1:
            raise RuntimeError() 

        # Otherwise return results
        return nk_fitted_param

    except RuntimeError:

        def mre_objective(param):
            y_pred = nk_model(xdata, *param, divide_by=1.)
            return np.mean(np.abs(ydata - y_pred) / ydata)
        
        nk_fitted_param = minimize(
            mre_objective, initial_param, method='Nelder-Mead').x

        return nk_fitted_param

def fit_evaluate_retention_model(
    y_true: tuple[float, ...],
    phi_start: tuple[np.ndarray, ...],
    phi_end: tuple[np.ndarray, ...],
    t_gradient: tuple[np.ndarray, ...],
    true_param: tuple[float, ...], 
    dwell_time: float = 0.614, 
    void_time: float = 0.258
) -> dict:

    y_true = np.array(y_true)
    phi_start = np.array(phi_start)
    phi_end = np.array(phi_end)
    t_gradient = np.array(t_gradient)

    pred_param = fit_retention_model(
        xdata=[phi_start, phi_end, t_gradient],
        ydata=y_true,
        dwell_time=dwell_time, 
        void_time=void_time)

    y_true_gra, phi_elution_true = nk_gradient_model(
        true_param, phi_start, phi_end, t_gradient, dwell_time, void_time)
    
    y_pred_gra, phi_elution_pred = nk_gradient_model(
        pred_param, phi_start, phi_end, t_gradient, dwell_time, void_time)

    isocratic_phi_values = np.array([
        0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    y_true_iso = nk_isocratic_model(true_param, isocratic_phi_values)
    
    y_pred_iso = nk_isocratic_model(pred_param, isocratic_phi_values)

    # Return a bunch of results, including relevant results for the reward
    # function but also other interesting results.
    return {
        "error_param": np.mean(np.abs(true_param - pred_param) / true_param),
        "error_iso":   np.mean(np.abs(y_true_iso - y_pred_iso) / y_true_iso),
        "error_gra":   np.mean(np.abs(y_true_gra - y_pred_gra) / y_true_gra),
        'y_true_elution': phi_elution_true,
        'y_pred_elution': phi_elution_pred,
        "pred_param":  pred_param,
        "true_param":  true_param,
        "y_true_gra":  y_true_gra,
        "y_pred_gra":  y_pred_gra,
        "y_true_iso":  y_true_iso,
        "y_pred_iso":  y_pred_iso,
    }

