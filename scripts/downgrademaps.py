import healpy as hp

def dg(fn):
    fn = 'maps/real/' + fn
    hmap = hp.ud_grade(hp.read_map(fn,field=(0,1,2)), 512, pess=True)
    fnout = fn.replace('2048','512dg')
    hp.write_map(fnout, hmap)

fn = ['HFI_SkyMap_217-ds1_2048_R2.02_full-ringhalf-1.fits',
      'HFI_SkyMap_217-ds1_2048_R2.02_full-ringhalf-2.fits',
      'HFI_SkyMap_217-ds2_2048_R2.02_full-ringhalf-1.fits',
      'HFI_SkyMap_217-ds2_2048_R2.02_full-ringhalf-2.fits',
      'HFI_SkyMap_217-ds1_2048_R2.02_halfmission-1.fits',
      'HFI_SkyMap_217-ds1_2048_R2.02_halfmission-2.fits',
      'HFI_SkyMap_217-ds2_2048_R2.02_halfmission-1.fits',
      'HFI_SkyMap_217-ds2_2048_R2.02_halfmission-2.fits',
      'HFI_SkyMap_353-ds1_2048_R2.02_full-ringhalf-1.fits',
      'HFI_SkyMap_353-ds1_2048_R2.02_full-ringhalf-2.fits',
      'HFI_SkyMap_353-ds2_2048_R2.02_full-ringhalf-1.fits',
      'HFI_SkyMap_353-ds2_2048_R2.02_full-ringhalf-2.fits',
      'HFI_SkyMap_353-ds1_2048_R2.02_halfmission-1.fits',
      'HFI_SkyMap_353-ds1_2048_R2.02_halfmission-2.fits',
      'HFI_SkyMap_353-ds2_2048_R2.02_halfmission-1.fits',
      'HFI_SkyMap_353-ds2_2048_R2.02_halfmission-2.fits',]

for f in fn:
    print f
    dg(f)


      
