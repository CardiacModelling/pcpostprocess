#
# Config for pcpostprocess.run_herg_qc
#

# Copy this config file to the top-level data directory and modify fields.

# Data output directory
savedir = './data'

# Save name for this set of data
saveID = 'EXPERIMENT_NAME'

# DataControl384 protocol output names to shorter names
# Protocols used for QC (usually the 'staircase' protocol)
D2S_QC = {
    'staircaseramp': 'staircaseramp'
}

# Additional protocols to export
D2S = {
}

