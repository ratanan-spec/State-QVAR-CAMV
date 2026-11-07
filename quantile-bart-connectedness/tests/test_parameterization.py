"""Scientific regression checks for the ALD conversion and the one-step target."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dgp',ROOT/'code/run_qbart_dgp_monte_carlo.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
report=[]
for tau in [.01,.05,.5,.95,.99]:
 sigma=.17;u=np.array([-3.,-.1,0.,.2,4.]);b=np.sqrt(tau*(1-tau))/sigma
 expected=np.log(tau*(1-tau)/sigma)-np.maximum(tau*u,(tau-1)*u)/sigma
 actual=pm.logp(pm.AsymmetricLaplace.dist(mu=0,b=b,q=tau),u).eval()
 np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)
 report.append({'test':'ALD check-loss equivalence','tau':tau,'max_error':float(np.max(abs(actual-expected))),'passed':True})
for dgp in ['linear','nonlinear','nonlinear_heavytail']:
 frame,_=mod.make_panel(300,20261107,dgp)
 for a in mod.ASSETS:
  np.testing.assert_allclose(frame[a+'_ret_lead1'].to_numpy()[:-1],frame[a+'_ret'].to_numpy()[1:])
 train,cutoff=mod.strict_train(frame,.8)
 assert train.target_date.max()<cutoff
 assert all(c.endswith('_ret') for c in mod.XCOLS)
 phi,tci,info=mod.oracle_network(train,.05,20,.5,10,dgp)
 np.testing.assert_allclose(phi.sum(1),1,atol=1e-12);assert np.all(phi>=0);assert 0<=tci<100
 report.append({'test':'one-step target and operator identities','dgp':dgp,'passed':True})
import json
(ROOT/'results/implementation_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
