"""Compact landscape figure from the same four measured example vectors."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

root=Path(__file__).resolve().parent
(root/'figures').mkdir(parents=True,exist_ok=True)
rev=root.parent/'revision_v2'
examples=json.loads((rev/'results/verified_examples.json').read_text(encoding='utf-8'))
samples=np.load(rev/'benchmark_17/X.npy',mmap_mode='r')
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':10,'pdf.fonttype':42})
fig,axes=plt.subplots(4,4,figsize=(8.5,5.1),sharex='col')
fig.subplots_adjust(left=.095,right=.925,top=.93,bottom=.12,wspace=.52,hspace=.17)
for row,item in enumerate(examples):
    x=samples[item['index']];t=np.arange(len(x))/6000
    axes[row,0].plot(t,x.real,lw=.45,label='I')
    axes[row,0].plot(t,x.imag,lw=.45,label='Q')
    axes[row,0].set(ylabel=item['mode'].replace('_','-'),xlim=(0,.3414),yticks=[-1,0,1])
    if row==0:axes[row,0].legend(loc='upper right',fontsize=6,ncol=2,handlelength=1,borderpad=.2)
    axes[row,1].scatter(x.real,x.imag,s=.45,alpha=.35)
    axes[row,1].set_aspect('equal',adjustable='box')
    axes[row,1].set(xticks=[-1,0,1],yticks=[-1,0,1])
    f,p=signal.welch(x,6000,nperseg=1024,return_onesided=False)
    axes[row,2].plot(np.fft.fftshift(f),10*np.log10(np.maximum(np.fft.fftshift(p),1e-20)),lw=.5)
    axes[row,2].set(xlim=(-3000,3000),ylim=(-85,-5),yticks=[-60,-20])
    # Show the actual occupied region without replacing the full-band overview.
    frequency=np.fft.fftshift(f);power=np.fft.fftshift(p)
    cdf=np.cumsum(power)/power.sum()
    lower,upper=frequency[np.searchsorted(cdf,[.005,.995])]
    center=(lower+upper)/2;half=max(60,(upper-lower)*.7)
    zoom=axes[row,2].inset_axes([.47,.46,.51,.48])
    zoom.plot(frequency,10*np.log10(np.maximum(power,1e-20)),lw=.65,color='#a42c35')
    zoom.set(xlim=(center-half,center+half),ylim=(-65,-5),yticks=[],
             xticks=[round(center-half),round(center+half)])
    zoom.tick_params(axis='x',labelsize=5.5,length=1.5,pad=.4)
    zoom.set_title('Local band (Hz)',fontsize=6,pad=1)
    zoom.set_facecolor('white')
    f,tt,z=signal.stft(x,6000,nperseg=512,noverlap=480,boundary=None,return_onesided=False)
    im=axes[row,3].pcolormesh(tt,np.fft.fftshift(f),20*np.log10(np.maximum(abs(np.fft.fftshift(z,axes=0)),1e-10)),
                              shading='auto',vmin=-55,vmax=0,rasterized=True)
    axes[row,3].set(ylim=(-700,700),yticks=[-500,0,500])
    for ax in axes[row]:ax.tick_params(labelsize=7.5,length=2,pad=1)
for ax,title in zip(axes[0],['I/Q waveforms','I/Q scatter','PSD (dB/Hz)','Spectrogram']):
    ax.set_title(title)
for ax,label in zip(axes[-1],['Time (s)','I','Frequency (Hz)','Time (s)']):
    ax.set_xlabel(label,labelpad=1)
axes[-1,0].set_xticks([0,.17,.34]);axes[-1,2].set_xticks([-3000,0,3000])
axes[-1,3].set_xticks([.1,.2,.3])
cax=fig.add_axes([.953,.12,.012,.79])
bar=fig.colorbar(im,cax=cax,ticks=[-50,-25,0])
bar.ax.tick_params(labelsize=7.5);bar.set_label('STFT amplitude (dB)',fontsize=8)
for suffix in ['pdf','png']:fig.savefig(root/'figures'/('figure5.'+suffix),dpi=220,bbox_inches='tight',pad_inches=.02)
plt.close(fig)
