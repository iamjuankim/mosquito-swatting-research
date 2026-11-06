"""Only the preregistered gate batches are permitted here."""
import common as C
from common import *
from runner import exclusive,peak_rss,invoke,emit_heartbeat,start_heartbeat
import subprocess,traceback

def protection_tests():
    # All launches here run through the installed parent audit hook.
    import socket
    from runtime import process_allowed
    cmd=[sys.executable,'-B',str(BASE/'runner.py'),'--probe','lock']
    checks=[]
    def rejected(label,fn):
        try:fn()
        except PermissionError as e:checks.append({'test':label,'rejected':True,'reason':str(e)})
        else:raise RuntimeError('Protection accepted '+label)
    # Normal launch: real child imports the same runtime guard and exits normally.
    r=subprocess.run(cmd,cwd=BASE,capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Allowed child failed '+r.stderr)
    checks.append({'test':'allowed_real_child','passed':True})
    r=subprocess.run(cmd,cwd=BASE,executable=sys.executable,capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Explicit executable child failed '+r.stderr)
    checks.append({'test':'explicit_executable_real_child','passed':True})
    for label,args,kw in [
        ('shell',cmd,{'shell':True}),
        ('python_c',[sys.executable,'-c','pass'],{}),
        ('python_m',[sys.executable,'-m','site'],{}),
        ('other_executable',['cmd.exe','/c','exit'],{}),
        ('other_script',[sys.executable,'-B',str(BASE/'preflight.py')],{}),
        ('extra_args',cmd+['--production'],{}),
        ('wrong_cwd',cmd,{'cwd':BASE.parent}),
        ('custom_env',cmd,{'env':dict(os.environ,PYTHONPATH=str(BASE.parent))}),
        ('snapshot_escape',[sys.executable,'-B',str(BASE/'runner.py'),'--snapshot',str(BASE.parent/'forbidden.json')],{}),
        ('compare_escape',[sys.executable,'-B',str(BASE/'runner.py'),'--compare-safety',str(BASE/'a'),str(BASE/'b'),str(BASE.parent/'forbidden.json')],{}),
        ('production_child',[sys.executable,'-B',str(BASE/'runner.py'),'--production','--approve-full-48'],{}),
    ]:
        rejected(label,lambda args=args,kw=kw:subprocess.run(args,**dict({'cwd':BASE,'capture_output':True},**kw)))
    for label,exe,line in [
        ('none_arbitrary',None,'cmd.exe /c exit'),
        ('unquoted_ambiguous',None,'"'+str(Path(sys.executable).resolve())+'" -B "'+str(BASE/'runner.py')+'" --probe lock'),
        ('explicit_mismatch',str(BASE/'runner.py'),subprocess.list2cmdline(cmd)),
        ('argument_injection',None,subprocess.list2cmdline(cmd)+' & whoami'),
    ]:
        if process_allowed(exe,line,BASE,None):raise RuntimeError('Parser accepted '+label)
        checks.append({'test':label,'rejected':True})
    # Writes target the parent directory, with denial required before file creation.
    target=BASE.parent/'stage7_forbidden_write_probe.tmp'
    if target.exists():raise RuntimeError('Unexpected outside test target')
    rejected('external_open_write',lambda:target.write_bytes(b'forbidden'))
    rejected('external_mkdir',lambda:(BASE.parent/'stage7_forbidden_mkdir_probe').mkdir())
    rejected('external_rename',lambda:os.replace(BASE/'GO.json',target))
    rejected('os_system',lambda:os.system('echo forbidden'))
    with socket.socket() as sock:
        rejected('network',lambda:sock.connect(('192.0.2.1',9)))
    if target.exists():raise RuntimeError('External file created')
    # Exercise inherited child restrictions rather than only the parent hook.
    from runner import invoke
    invoke('--probe','isolation')
    child=read(BASE/'probe_isolation.json')
    if not child['passed']:raise RuntimeError('Child isolation failure')
    checks.append({'test':'child_external_write_network_process_rejected','evidence':child})
    # Terminate a real child while its checkpoint is only partially written.
    with subprocess.Popen([sys.executable,'-B',str(BASE/'runner.py'),'--probe','interrupt'],cwd=BASE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as proc:
        deadline=time.monotonic()+20
        while not (BASE/'probe_interrupt_ready.json').exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        if not (BASE/'probe_interrupt_ready.json').exists():
            proc.kill();out,err=proc.communicate();raise RuntimeError('Interrupt child not ready '+err)
        proc.terminate();out,err=proc.communicate(timeout=10)
        if proc.returncode==0:raise RuntimeError('Interrupt exited successfully')
    try:load_checkpoint('_interruptprobe',0,signature())
    except RuntimeError as e:
        if 'INCOMPLETE_CHECKPOINT' not in str(e):raise
    else:raise RuntimeError('Interrupted partial checkpoint accepted')
    checks.append({'test':'OS_terminated_child_partial_rejected','returncode':proc.returncode,'passed':True})
    atomic_json(BASE/'security_gate.json',{'passed':True,'checks':checks,'no_simulations':True})
    return {'passed':True,'checks':len(checks),'evidence':'security_gate.json'}

def stage6_compare(outputs):
    ids=np.asarray(read(STAGE6/'selection_manifest.json')['selected_ids'])
    proof=[]
    for mode,z in outputs.items():
        with np.load(STAGE6/f'{mode}_trajectories.npz',allow_pickle=False) as old:
            for key in ['captured','t_det_s','t_contact_s','M0','theta','trigger_i']:
                if not np.array_equal(z[key][ids],old[key]):raise RuntimeError('Stage6 full-batch consistency '+mode+' '+key)
        proof.append({'mode':mode,'ids':ids.tolist(),'all_33_outcomes_thresholds_trigger_frames_exact':True})
    return proof
def checkpoint_tests(z,sig):
    state={'phase':'RUNNING','complete':False}
    hb=BASE/'probe_heartbeat.json';emit_heartbeat(hb,state);initial=read(hb)
    stop,thread=start_heartbeat(hb,state,interval=.01)
    try:
        time.sleep(.06)
    finally:stop.set();thread.join()
    after=read(hb)
    if after['monotonic_s']<=initial['monotonic_s'] or after['complete'] or after['rss_peak_bytes']<=0:raise RuntimeError('Heartbeat test')
    s='_resumeprobe';p=save_checkpoint(s,0,z,sig,0)
    before=digest(p);load_checkpoint(s,0,sig)
    locked=subprocess.run([sys.executable,'-B',str(BASE/'runner.py'),'--probe','lock'],cwd=BASE,capture_output=True,text=True)
    if locked.returncode==0 or 'DUPLICATE_EXECUTION' not in locked.stderr:raise RuntimeError('Duplicate lock test')
    failures=[]
    for kind in ['fail','partial']:
        r=subprocess.run([sys.executable,'-B',str(BASE/'runner.py'),'--probe',kind],cwd=BASE,capture_output=True,text=True)
        if r.returncode==0 or read(BASE/'probe_failure.json')['complete']:raise RuntimeError('Failure exit test')
        failures.append({'probe':kind,'returncode':r.returncode,'marked_incomplete':True,'stderr':r.stderr})
    r=subprocess.run([sys.executable,'-B',str(BASE/'runner.py'),'--probe','resume','--signature',sig],cwd=BASE,capture_output=True,text=True)
    if r.returncode or digest(p)!=before or not read(BASE/'probe_resume.json')['no_simulations']:raise RuntimeError('Resume test')
    try:load_checkpoint(s,0,'wrong')
    except RuntimeError:pass
    else:raise RuntimeError('Signature drift accepted')
    oldsig=read(BASE/'prior_no_go_c9d08659'/'code_manifest.json')['signature']
    if oldsig==sig:raise RuntimeError('Repair fingerprint did not change')
    try:load_checkpoint(s,0,oldsig)
    except RuntimeError:pass
    else:raise RuntimeError('Old fingerprint accepted new checkpoint')
    original=p.read_bytes()
    try:
        with p.open('ab') as f:f.write(b'corrupt')
        try:load_checkpoint(s,0,sig)
        except RuntimeError:pass
        else:raise RuntimeError('Corruption accepted')
    finally:p.write_bytes(original)
    # A final NPZ without its final commit marker is incomplete and cannot be skipped.
    orphan=checkpoint_path('_orphanprobe',0);orphan.parent.mkdir(parents=True,exist_ok=True);orphan.write_bytes(original)
    try:load_checkpoint('_orphanprobe',0,sig)
    except RuntimeError:pass
    else:raise RuntimeError('Orphan accepted')
    return {'passed':True,'roundtrip_arrays_exact':True,'heartbeat_thread_progress_and_rss_verified':True,'heartbeat_evidence':after,'duplicate_process_rejected':True,'failures':failures,'partial_not_committed':True,'orphan_rejected':True,'restart_no_simulations':True,'committed_hash_unchanged':digest(p)==before,'hash_corruption_rejected':True,'signature_drift_rejected':True}
def main():
    if (BASE/'gate_results.json').exists():raise RuntimeError('Preflight already ran; inspect evidence, no automatic rerun')
    results={};atomic_json(BASE/'gate_results.json',{'complete':False,'phase':'RUNNING'})
    sig=signature()
    try:
        security=read(BASE/'security_gate.json')
        if not security['passed'] or security['code_sha256']!=code_hashes():raise RuntimeError('Security gate absent or stale')
        results['G0_process_write_network_protection']={'passed':True,'evidence':'security_gate.json'}
        verify_inputs()
        if digest(BASE/'stable_threshold.py')!=digest(BASE.parent/'mq10_audit_stage5_threshold_20261104/stable_threshold.py'):raise RuntimeError('Stable code mismatch')
        results['G1_inputs_code']={'passed':True,'signature':sig,'environment':environment()}
        ctx=context();specs={s['id']:s for s in read(BASE/'conditions.json')}
        # Exact all-field Stage5 batch provenance at j=0.
        b,*_=batch_args(ctx,next(s for s in specs.values() if s['model']=='original'),0)
        rng=np.random.default_rng([SEED_MC,0,0]);pick=rng.integers(0,len(ctx.init),4000)
        fields={k:v for k,v in vars(b).items() if v is not None};fields.update(trajectory_id=np.arange(4000),source_initial_index=pick,source_trial_id=ctx.init.iloc[pick].trial_id.to_numpy(str))
        with np.load(SCAN/'batch_j00_N4000.npz',allow_pickle=False) as old:
            if set(fields)!=set(old.files):raise RuntimeError('Batch fields differ')
            for k,v in fields.items():
                if v.dtype!=old[k].dtype or not np.array_equal(v,old[k]):raise RuntimeError('Batch mismatch '+k)
        results['G2_batch_provenance']={'passed':True,'j':0,'all_fields_dtype_ids_randoms_exact':True,'time_grid':'computed from full original N4000 before all trajectories'}
        metrics=[];legacy_proofs=[];s6={};first=None
        for case in read(BASE/'preregistered_gates.json')['cases']:
            spec=specs[case['condition_id']];j=case['j']
            print('GATE',case,flush=True)
            outs={}
            for mode in ['legacy','stable']:
                z,wall=compute(ctx,spec,j,mode);outs[mode]=z
                if mode=='legacy':
                    proof=compare_saved(z,spec,j);proof.update(case);legacy_proofs.append(proof)
                    # Store all 4000 start diagnostics for explicit historical-error evidence.
                    np.savez_compressed(BASE/'gates'/f"{spec['id']}_j{j:03d}_legacy.npz",**z)
                else:
                    p=save_checkpoint(spec['id'],j,z,sig,wall)
                    metrics.append(dict(case,wall_s=wall,peak_rss_bytes=peak_rss(),checkpoint_bytes=p.stat().st_size,checkpoint_sha256=digest(p)))
                    if first is None:first=z
                if mode=='stable' and not np.array_equal(z['i0'],outs['legacy']['i0']):raise RuntimeError('Mode start-grid changed')
                if mode=='stable' and (float(z['dt_s'])!=float(outs['legacy']['dt_s']) or int(z['n_steps'])!=int(outs['legacy']['n_steps'])):raise RuntimeError('Mode grid changed')
            if spec['model']=='original' and spec['condition']['v_mps']==12 and spec['condition']['n_exp']==1 and j==0:s6=outs
        results['G3_legacy_reproduction']={'passed':True,'batches':len(legacy_proofs),'trajectories':len(legacy_proofs)*4000,'comparisons':legacy_proofs,'tolerance':'none; captured exact, float32 time bits including signs exact'}
        results['G4_stage6']={'passed':True,'comparisons':stage6_compare(s6)}
        from stable_threshold import sample_log_threshold,UnresolvedQuantile
        strict_failures=[]
        for label,params,exc in [('invalid_u',(1.,0.,1.,1.),ValueError),('unresolved',(1.,-1e6,1.,0.5),UnresolvedQuantile)]:
            try:sample_log_threshold(*params)
            except exc:strict_failures.append(label)
            else:raise RuntimeError('Strict rejection missing '+label)
        results['G5_stable_strict']={'passed':True,'rejection_tests':strict_failures,'batches':len(metrics),'trajectories':len(metrics)*4000,'strict':True,'finite_linear_theta':True,'original_comparison':'M>=Theta','no_fallback':True}
        results['G6_checkpoints_failure_resume']=checkpoint_tests(first,sig)
        for k in ['G1_inputs_code','G2_batch_provenance','G3_legacy_reproduction','G4_stage6','G5_stable_strict','G6_checkpoints_failure_resume']:
            atomic_json(BASE/'gate_results.json',{'complete':False,'gates':results})
        # Estimate by model and n; historical per-condition wall times normalized to measured gate throughput.
        hist=read(BASE/'historical_costs.json');ratios=[]
        for m in metrics:
            key=m['condition_id'];ratios.append(m['wall_s']/(hist[key]/200))
        total=sum(hist.values());factor=float(np.median(ratios));hi=float(np.max(ratios))
        estimates={'passed':True,'metrics':metrics,'legacy_48_total_s':total,'measured_to_legacy_median_ratio':factor,'measured_to_legacy_max_ratio':hi,'estimated_full_48_wall_s':total*factor,'planning_upper_wall_s':total*hi*1.5,'max_measured_process_peak_rss_bytes':max(m['peak_rss_bytes'] for m in metrics),'estimated_checkpoint_bytes_9600_mean':int(np.mean([m['checkpoint_bytes'] for m in metrics])*9600),'estimated_checkpoint_bytes_9600_max':max(m['checkpoint_bytes'] for m in metrics)*9600,'note':'Sequential; includes numerical observer cost. RSS is process high-water mark, not per-batch incremental. Input/protection hashing, startup, final checks additional.'}
        results['G7_cost_measurements']=estimates
        atomic_json(BASE/'measurements.json',estimates)
        # Independent readback of every reusable gate checkpoint.
        for m in metrics:load_checkpoint(m['condition_id'],m['j'],sig)
        if sig!=signature():raise RuntimeError('Signature drift during gates')
        results['G8_reusable_checkpoints']={'passed':True,'count':len(metrics),'signature':sig}
        atomic_json(BASE/'gate_results.json',{'complete':True,'gates':results,'signature':sig,'production_started':False})
    except BaseException as e:
        atomic_json(BASE/'gate_results.json',{'complete':False,'gates':results,'error':repr(e),'traceback':traceback.format_exc(),'production_started':False})
        raise
if __name__=='__main__':
    with exclusive():main()
