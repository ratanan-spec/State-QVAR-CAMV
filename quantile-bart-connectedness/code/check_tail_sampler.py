"""Controlled training-only comparison of default and fuller BART tree sweeps."""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,pymc as pm,pymc_bart as pmb,arviz as az
from scipy.optimize import linprog
HERE=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dgp',HERE/'run_qbart_dgp_monte_carlo.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--draws',type=int,default=6000);p.add_argument('--tune',type=int,default=3000);p.add_argument('--particles',type=int,default=30);p.add_argument('--batch',type=float,default=1.0);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
frame,_=m.make_panel(600,20261110,'linear');train,_=m.strict_train(frame,.8);Xraw=train[m.XCOLS].to_numpy();X=(Xraw-Xraw.mean(0))/Xraw.std(0,ddof=1);yr=train[m.YCOLS[0]].to_numpy();y=(yr-yr.mean())/yr.std(ddof=1);xd=np.c_[np.ones(len(X)),X];n,pcols=xd.shape;tau=.05
qr=linprog(np.r_[np.zeros(pcols),np.full(n,tau),np.full(n,1-tau)],A_eq=np.c_[xd,np.eye(n),-np.eye(n)],b_eq=y,bounds=[(None,None)]*pcols+[(0,None)]*(2*n),method='highs');assert qr.success
base=xd@qr.x[:pcols];r=y-base;sigma=max(np.maximum(tau*r,(tau-1)*r).mean(),.05);mus=[]
for c in range(4):
 with pm.Model():
  mu=pmb.BART('mu',X,r,m=20,alpha=.95,beta=2.);pm.AsymmetricLaplace('obs',mu=base+mu,b=np.sqrt(tau*(1-tau))/sigma,q=tau,observed=y);step=pmb.PGBART(vars=[mu],num_particles=a.particles,batch=(a.batch,a.batch));i=pm.sample(tune=a.tune,draws=a.draws,chains=1,cores=1,step=step,random_seed=20261160+c,progressbar=False,compute_convergence_checks=False)
 mus.append(np.asarray(i.posterior.to_dataset()['mu'].values[0]));print('completed chain',c+1,flush=True)
idata=az.from_dict({'posterior':{'mu':np.stack(mus)}});rh=az.rhat(idata,var_names=['mu']).to_dataset()['mu'].values;ess=az.ess(idata,var_names=['mu'],method='bulk').to_dataset()['mu'].values
res={'target':'BTC synthetic lower-tail surface only; no network recovery claim','tau':tau,'draws':a.draws,'tune':a.tune,'chains':4,'particles':a.particles,'batch':a.batch,'max_rhat':float(rh.max()),'rhat_p95':float(np.quantile(rh,.95)),'min_bulk_ess':float(ess.min()),'surface_gate_pass':bool(rh.max()<1.01 and ess.min()>=400)};(out/'sampler_diagnostic.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
