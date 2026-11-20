"""Shared steady/transient temperature and refracted-ray field figures."""
from pathlib import Path
from thermal_opd.run_io import figure_directory
import matplotlib.pyplot as plt
import numpy as np


def save_temperature_opd_figure(path,r,phi,temperature,x,y,mask,opd,*,title,pv_m,continuous_pv_m=None,figsize=(11,10)):
    """Same 3 temperature planes + entry OPD, 61 levels and 180 dpi.

    Three temperatures share one scale within a snapshot. OPD is displayed
    relative to that aperture's minimum; the saved raw field is not altered.
    """
    plot_phi = np.append(phi,2*np.pi)
    tx,ty = r[:,None]*np.cos(plot_phi)*1e3,r[:,None]*np.sin(plot_phi)*1e3
    fig,axes = plt.subplots(2,2,figsize=figsize,layout='constrained')
    low,high = float(np.min(temperature)),float(np.max(temperature))
    if high<=low:
        high = low+1e-12
    levels = np.linspace(low,high,61)
    for ax,field,name in zip(axes.flat,temperature,('Front: z=0','Mid-plane: z=L/2','Rear: z=L')):
        contour = ax.contourf(tx,ty,np.column_stack((field,field[:,0])),levels=levels,cmap='inferno')
        ax.set_title(name)
        fig.colorbar(contour,ax=ax,label='Temperature rise [K]',shrink=.85)
    optical = np.where(mask,(opd-np.min(opd[mask]))*1e9,np.nan)
    contour = axes.flat[-1].contourf(x*1e3,y*1e3,optical,levels=61,cmap='viridis')
    optical_title = f'OPD PV [nm]\nSampled: {pv_m*1e9:.4f}'
    if continuous_pv_m is not None:
        optical_title += f'\nContinuous: {continuous_pv_m*1e9:.4f}'
    axes.flat[-1].set_title(optical_title)
    fig.colorbar(contour,ax=axes.flat[-1],label='OPD minus aperture minimum [nm]',shrink=.85)
    for ax in axes.flat:
        ax.set_aspect('equal');ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
    axes.flat[-1].set_xlabel('Entry x [mm]');axes.flat[-1].set_ylabel('Entry y [mm]')
    fig.suptitle(title)
    fig.savefig(Path(path),dpi=180)
    plt.close(fig)

def save_evolution(directory,f,last):
    indices = range(1,last)
    fig,axes = plt.subplots(2,len(indices),figsize=(19,7),layout='constrained')
    r,phi = f['temperature_r_m'],np.r_[f['temperature_phi_rad'],2*np.pi]
    tx,ty = r[:,None]*np.cos(phi)*1e3,r[:,None]*np.sin(phi)*1e3
    temperature_high = np.max(f['temperature_K'][list(indices),0])
    opd_high = max(np.ptp(f['opd_m'][j][f['aperture_mask']])*1e9 for j in indices)
    for column,j in enumerate(indices):
        T = f['temperature_K'][j,0]
        contour = axes[0,column].contourf(tx,ty,np.column_stack((T,T[:,0])),levels=np.linspace(0,temperature_high,61),cmap='inferno',extend='min')
        O = f['opd_m'][j]
        optical = (O-np.min(O[f['aperture_mask']]))*1e9
        ray = axes[1,column].contourf(f['opd_x_m']*1e3,f['opd_y_m']*1e3,optical,levels=np.linspace(0,opd_high,61),cmap='viridis')
        axes[0,column].set_title(f't={f["times_s"][j]:g} s')
        for ax in axes[:,column]:
            ax.set_aspect('equal');ax.set_xlabel('x [mm]');ax.set_ylabel('y [mm]')
    fig.colorbar(contour,ax=axes[0,:],label='Front temperature rise [K]')
    fig.colorbar(ray,ax=axes[1,:],label='OPD minus aperture minimum [nm]')
    fig.suptitle('Transient evolution: fixed scales across time within each row')
    fig.savefig(figure_directory(directory)/'transient_evolution.png',dpi=180);plt.close(fig)


def save_history(directory,t,T,pv,steady_T,steady_pv,snapshots):
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
    early = t<=75.
    colors = ('tab:red','tab:orange','tab:blue')
    for i,(name,color) in enumerate(zip(('front','middle','rear'),colors)):
        axes[0,0].plot(t[early],T[early,i,0],color=color,label=f'center {name}')
        axes[0,1].semilogx(t[1:],T[1:,i,0],color=color,label=f'center {name}')
        axes[0,1].axhline(steady_T[i,0],color=color,linestyle='--',alpha=.6)
    axes[1,0].plot(t[early],pv[early]*1e9,label='201x201 sampled PV')
    rows = [row for row in snapshots if 0<row['time_s']<=75]
    axes[1,0].scatter([row['time_s'] for row in rows],[row['continuous_pv_m']*1e9 for row in rows],
                      color='black',s=20,label='continuous numerical PV')
    axes[1,1].semilogx(t[1:],pv[1:]*1e9,label='201x201 sampled PV')
    axes[1,1].axhline(steady_pv*1e9,color='black',linestyle='--',label='same modes steady')
    for i,ax in enumerate(axes.flat):
        ax.set_xlabel('Time [s]');ax.set_ylabel('Temperature rise [K]' if i<2 else 'OPD PV [nm]')
        ax.grid(alpha=.25);ax.legend()
    axes[0,0].set_title('Early center temperature history')
    axes[0,1].set_title('Approach to steady temperature')
    axes[1,0].set_title('Early refracted-ray OPD history')
    axes[1,1].set_title('Approach to steady OPD')
    fig.savefig(figure_directory(directory)/'transient_history.png',dpi=180);plt.close(fig)


def save_early_depths(directory,f):
    """Separate scales expose the extremely small early rear temperature."""
    r,phi = f['temperature_r_m'],np.r_[f['temperature_phi_rad'],2*np.pi]
    x,y = r[:,None]*np.cos(phi)*1e3,r[:,None]*np.sin(phi)*1e3
    fig,axes = plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,j in enumerate((1,2)):
        for plane,name in enumerate(('front','middle','rear')):
            T = f['temperature_K'][j,plane]
            contour = axes[row,plane].contourf(x,y,np.column_stack((T,T[:,0])),levels=61,cmap='inferno')
            axes[row,plane].set_title(f't={f["times_s"][j]:g} s, {name}')
            axes[row,plane].set_aspect('equal');axes[row,plane].set_xlabel('x [mm]');axes[row,plane].set_ylabel('y [mm]')
            fig.colorbar(contour,ax=axes[row,plane],label='Temperature rise [K]',shrink=.85)
    fig.suptitle('Early depth response: each panel uses its own temperature scale')
    fig.savefig(figure_directory(directory)/'transient_early_depths.png',dpi=180);plt.close(fig)


def save_modal_contribution_figure(path,times,orders,shares,order_pv,total_pv):
    """Thermal order norm shares and their individual optical contributions."""
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for i,m in enumerate(orders):
        axes[0].semilogx(times[1:],shares[i,1:,0],marker='o',label=f'm={m}')
        axes[1].semilogx(times[1:],order_pv[i,1:]*1e9,marker='o',label=f'thermal m={m}')
    axes[1].semilogx(times[1:],total_pv[1:]*1e9,'k--',label='total')
    axes[0].set_ylabel('Front temperature L2 squared share')
    axes[1].set_ylabel('Single-order sampled OPD PV [nm]')
    for ax in axes:
        ax.set_xlabel('Time [s]');ax.grid(alpha=.25);ax.legend()
    fig.savefig(path,dpi=180);plt.close(fig)


def replot_saved_case(results,*,output_dir=None):
    """Render existing field arrays into a new run; never re-solve physics."""
    import json
    from thermal_opd.run_io import create_run,data_directory,write_summary,provenance
    source=Path(results).resolve()
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status','COMPLETE')!='COMPLETE':
        raise ValueError('Cannot replot an incomplete run')
    kind=summary.get('kind')
    if kind not in ('complete_representative_steady_refracted_ray_case','transient_fine_fields'):
        raise ValueError('Replot supports representative steady and fine transient field runs')
    root=Path(__file__).resolve().parents[2]
    directory=create_run(root,'replot',output_dir)
    figures=figure_directory(directory)
    names=[]
    with np.load(data_directory(source)/'fields.npz',allow_pickle=False) as f:
        r,phi=f['temperature_r_m'],f['temperature_phi_rad']
        x,y=f['opd_x_m'],f['opd_y_m']
        if x.ndim==y.ndim==1:
            x,y=np.meshgrid(x,y)
        mask=f['aperture_mask']
        if kind=='complete_representative_steady_refracted_ray_case':
            name='steady_case.png'
            save_temperature_opd_figure(figures/name,r,phi,f['temperature_rise_K'],x,y,mask,f['opd_m'],
                title=f"Steady: {summary['incidence_angle_deg']:g} deg",pv_m=np.ptp(f['opd_m'][mask]),
                continuous_pv_m=summary.get('opd_continuous_numerical_extrema_m',{}).get('pv'))
            names.append(name)
        else:
            for j,t in enumerate(f['times_s']):
                if t==0:
                    continue
                name=f'transient_t{t:.6g}s.png'
                save_temperature_opd_figure(figures/name,r,phi,f['temperature_K'][j],x,y,mask,f['opd_m'][j],
                    title=f"Transient: {summary['angle_deg']:g} deg, t={t:g}s",pv_m=np.ptp(f['opd_m'][j,mask]),
                    continuous_pv_m=summary['snapshots'][j].get('continuous_pv_m'))
                names.append(name)
            save_temperature_opd_figure(figures/'steady_limit.png',r,phi,f['steady_temperature_K'],x,y,mask,f['steady_opd_m'],
                title='Same-mode steady limit',pv_m=np.ptp(f['steady_opd_m'][mask]),
                continuous_pv_m=summary.get('steady_extrema',{}).get('pv'))
            save_evolution(figures,f,len(f['times_s'])-1)
            save_early_depths(figures,f)
            save_modal_contribution_figure(figures/'transient_modes.png',f['times_s'],f['orders'],
                f['temperature_plane_l2_squared_shares'],f['order_sampled_opd_pv_m'],f['sampled_pv_m'])
            names+=['steady_limit.png','transient_evolution.png','transient_early_depths.png','transient_modes.png']
            with np.load(data_directory(source)/'history.npz',allow_pickle=False) as h:
                save_history(figures,h['times_s'],h['temperature_K'],h['sampled_pv_m'],h['steady_temperature_K'],
                             h['steady_sampled_pv_m'],summary['snapshots'])
            names.append('transient_history.png')
    write_summary(directory,dict(kind='saved_field_replot',status='COMPLETE',source_run=str(source),figures=names,
        physical_parameters_SI=summary.get('physical_parameters_SI'),numerical_settings=summary.get('numerical_settings'),
        provenance=provenance(root,__file__),verification_references=[str(source/'verification.json'),str(source/'check_report.json')],
        note='Rendering only; source evidence is referenced, no new scientific verification was executed.'))
    return directory

def save_transient_diagnostic_figure(directory,fields):
    times,orders = fields['times_s'],fields['orders']
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for i,m in enumerate(orders):
        axes[0].plot(times,fields['temperature_plane_l2_squared_shares'][i,:,0],marker='o',label=f'm={m}')
        axes[1].plot(times,fields['order_sampled_opd_pv_m'][i]*1e9,marker='o',label=f'thermal m={m}')
    axes[0].set_ylabel('Front thermal L2 squared share');axes[0].legend()
    axes[1].plot(times,np.ptp(fields['opd_sa_m'],axis=-1)*1e9,'k--',label='total')
    axes[1].set_ylabel('Single-order sampled OPD PV [nm]');axes[1].legend()
    for i,values in enumerate(fields['opd_fem_m']):
        axes[2].plot(times,np.ptp(values,axis=-1)*1e9,marker='.',label=f'FEM run {i}')
    axes[2].plot(times,np.ptp(fields['opd_sa_fem_common_m'],axis=-1)*1e9,'k--',label='SA')
    axes[2].set_ylabel('Sampled OPD PV on common aperture [nm]');axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Time [s]');ax.grid(alpha=.25)
    fig.savefig(figure_directory(directory)/'transient_modes.png',dpi=160)
    plt.close(fig)
