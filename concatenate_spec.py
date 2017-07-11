import decorr
LR = decorr.getLR()

for k,val in enumerate(LR):
    decorr.concatenate_spec('spec/gaussian/synch_therm_{:s}_mcplusexcess_noise_[0-9]???.pickle'.format(val))
    decorr.concatenate_spec('spec/gaussian/synch_therm_{:s}_qucov_noise_[0-9]???.pickle'.format(val))
    decorr.concatenate_spec('spec/dust0_tophat/synch_therm_{:s}_mcplusexcess_noise_[0-9]???.pickle'.format(val))
    decorr.concatenate_spec('spec/dust1_tophat/synch_therm_{:s}_mcplusexcess_noise_[0-9]???.pickle'.format(val))
    decorr.concatenate_spec('spec/dust2_tophat/synch_therm_{:s}_mcplusexcess_noise_[0-9]???.pickle'.format(val))
