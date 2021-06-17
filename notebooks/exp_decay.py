# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os

# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.gp.util import plot_gp_dist
import arviz as az
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')


# %%
Ts = [1.5, 4.5]
As = [0.7, 0.3]

n_samples = 128
edc = np.zeros(n_samples)
times = np.linspace(0, 3, n_samples)
for T_i, A_i in zip(Ts, As):
    edc += A_i*np.exp(-13.8/T_i*times)

edc = 10**((10*np.log10(edc) + np.random.normal(0, 0.25, n_samples))/10)


# %%
plt.plot(times, 10*np.log10(edc))

# %% [markdown]
# ## Gaussian Process
# export CPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include

# %%

rho_val = np.round(np.abs(times[0]-times[1])*10, decimals=2)

with pm.Model() as gp_edc_model:

    # Lengthscale
    rho = pm.HalfCauchy('rho', rho_val)
    eta = pm.HalfCauchy('eta', 25)

    M = pm.gp.mean.Linear(coeffs=1/Ts[0])
    K = (eta**2) * pm.gp.cov.ExpQuad(1, rho)

    sigma = pm.Normal('sigma', 1)

    recruit_gp = pm.gp.Marginal(mean_func=M, cov_func=K)
    recruit_gp.marginal_likelihood(
        'edc', X=times.reshape(-1, 1),
        y=10*np.log10(np.abs(edc)),
        noise=sigma)


# %%
with gp_edc_model:
    trace = pm.sample(1000, tune=1000, init='advi', cores=1)

# %%
az.plot_trace(trace, var_names=['rho', 'eta', 'sigma'])

# %%
with gp_edc_model:
    fit = pm.fit(20000)

trace = fit.sample(draws=1000)
# %%
az.plot_trace(trace, var_names=['rho', 'eta', 'sigma'])


# %%

times_pred = np.linspace(0, times[-1], 3*times.shape[-1])
dt = np.abs(np.diff(times_pred)[0])

with gp_edc_model:
    edc_pred = recruit_gp.conditional("edc_pred", times_pred.reshape(-1, 1))
    gp_edc_samples = pm.sample_posterior_predictive(
        trace, var_names=['edc_pred'], samples=500)

# %%
from pymc3.gp.util import plot_gp_dist
fig, ax = plt.subplots(figsize=(8, 6))
plot_gp_dist(ax, gp_edc_samples['edc_pred'], times_pred)
plt.plot(times, 10*np.log10(edc))
ax.plot(
    times_pred,
    np.mean(gp_edc_samples['edc_pred'], axis=0),
    label='mean', color='green', linestyle='--')
# salmon_data.plot.scatter(x='spawners', y='recruits', c='k', s=50, ax=ax)
# ax.set_ylim(0, 350);


# %%

fig, ax = plt.subplots(figsize=(8, 6))
plot_gp_dist(ax, gp_edc_samples['edc_pred'], times_pred)
ax.plot(
    times_pred,
    np.mean(gp_edc_samples['edc_pred'], axis=0),
    label='mean', color='green', linestyle='--')
# ax.set_ylim(0, 350);

# np.mean(gp_edc_samples['edc_pred'], axis=0) - 10*np.log10(np.abs(edc))

# %%
# fig, ax = plt.subplots(figsize=(8, 6))

# plt.plot(np.gradient(np.mean(gp_edc_samples['edc_pred'], axis=0), np.diff(times_pred)[0], edge_order=2))
# plt.plot(np.gradient(np.mean(gp_edc_samples['edc_pred'], axis=0)))
# %%
grad = np.gradient(np.mean(gp_edc_samples['edc_pred'], axis=0), dt, edge_order=1)

# %%
plt.plot(times_pred, np.abs(1/grad*60))
ax = plt.gca()
ax.set_ylim(0, 5)
# %%
