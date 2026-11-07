"""Repeated one-step BART recovery with failure accounting and resumable cells.
This executes real fits. A protocol file or generate-only run is not validation.
"""
import argparse,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dgp',HERE/'run_qbart_dgp_monte_carlo.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replicates',type=int,default=200);p.add_argument('--observations',type=int,default=1000);p.add_argument('--tune',type=int,default=6000);p.add_argument('--draws',type=int,default=6000);p.add_argument('--seed',type=int,default=20261107);p.add_argument('--dgp',nargs='+',default=['linear','nonlinear','nonlinear_heavytail']);p.add_argument('--tau',nargs='+',type=float,default=[.05,.5,.95]);a=p.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cells=[];edges=[]
 for dgp in a.dgp:
  for rep in range(a.replicates):
   seed=a.seed+10007*rep;panel,_=m.make_panel(a.observations,seed,dgp);train,cut=m.strict_train(panel,.8);data=out/dgp/f'rep_{rep:04d}.csv';data.parent.mkdir(parents=True,exist_ok=True);panel.to_csv(data,index=False)
   for tau in a.tau:
    cell=out/dgp/f'rep_{rep:04d}_q{tau:.2f}';cell.mkdir(exist_ok=True);status=cell/'validation_status.json'
    phi,truth,info=m.oracle_network(train,tau,40,.5,10,dgp)
    if not status.exists():
     cmd=[sys.executable,str(HERE/'fit_qbart_connectedness_strict.py'),'--data',str(data),'--out',str(cell),'--tau',str(tau),'--chains','4','--tune',str(a.tune),'--draws',str(a.draws),'--states','40','--seed',str(seed+int(tau*1000))]
     with (cell/'fit.log').open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
     status.write_text(json.dumps({'returncode':r.returncode,'seed':seed,'dgp':dgp,'replicate':rep,'tau':tau}))
    st=json.loads(status.read_text());row={'dgp':dgp,'replicate':rep,'tau':tau,'oracle_tci':truth,**st}
    if st['returncode']!=0:row.update(computed=False,convergence_pass=False);cells.append(row);continue
    draws=np.load(cell/'posterior_response_network_draws.npz');e=draws['edges'];t=draws['tci'];nd=pd.read_csv(cell/'network_diagnostics.csv').iloc[0];sd=pd.read_csv(cell/'surface_coordinate_diagnostics.csv')
    passed=bool(nd.edge_rhat_max<1.01 and nd.tci_rhat<1.01 and sd.rhat.max()<1.01 and nd.edge_bulk_ess_min>=400)
    lo,hi=np.quantile(t,[.025,.975]);row.update(computed=True,convergence_pass=passed,tci_error=float(t.mean()-truth),tci_squared_error=float((t.mean()-truth)**2),tci_coverage=bool(lo<=truth<=hi),tci_width=float(hi-lo),max_edge_rhat=float(nd.edge_rhat_max),max_surface_rhat=float(sd.rhat.max()));cells.append(row)
    el,eh=np.quantile(e,[.025,.975],axis=(0,1));emean=e.mean((0,1))
    for i in range(5):
     for j in range(5):edges.append({'dgp':dgp,'replicate':rep,'tau':tau,'receiver':i,'sender':j,'error':float(emean[i,j]-phi[i,j]),'covered':bool(el[i,j]<=phi[i,j]<=eh[i,j]),'width':float(eh[i,j]-el[i,j]),'convergence_pass':passed})
    pd.DataFrame(cells).to_csv(out/'cells.csv',index=False);pd.DataFrame(edges).to_csv(out/'edge_errors.csv',index=False)
    print(dgp,rep,tau,'passed',passed,flush=True)
 df=pd.DataFrame(cells);df.to_csv(out/'cells.csv',index=False);pd.DataFrame(edges).to_csv(out/'edge_errors.csv',index=False)
 # Unconverged runs are retained and flagged; no success-only primary estimate.
 aggregates=[]
 for (dgp,tau),g in df.groupby(['dgp','tau']):
  ok=g[g.computed];n=len(ok);pcover=float(ok.tci_coverage.mean()) if n else np.nan
  aggregates.append({'dgp':dgp,'tau':tau,'requested':a.replicates,'computed':n,'convergence_passes':int(g.convergence_pass.sum()),'bias_diagnostic':float(ok.tci_error.mean()) if n else np.nan,'rmse_diagnostic':float(np.sqrt(ok.tci_squared_error.mean())) if n else np.nan,'coverage_diagnostic':pcover,'coverage_mcse':float(np.sqrt(pcover*(1-pcover)/n)) if n else np.nan,'publication_gate_pass':bool(n==a.replicates and g.convergence_pass.all()),'interpretation':'Diagnostic only unless all computation/convergence gates pass; interval calibration is an empirical outcome, not assumed.'})
 pd.DataFrame(aggregates).to_csv(out/'aggregate.csv',index=False);(out/'protocol.json').write_text(json.dumps(vars(a),indent=2))
if __name__=='__main__':main()
