def redshift_to_mpc(z):
    
    H0=70 #Hubble constant in km/s/mpc
    c=300000 #speed of light in km/s

    return (z * c) / H0

def mpc_to_lightyears(mpc):
    return mpc * 3.26e6   #mpc * lightyears   = 3.26 10^6 light-years

# 1 pc   = 3.26 light-years
# 1 Mpc  = 3.26 million light-years
# Parsec = astronomy unit  
# Megaparsec = large-scale universe distance  
# Light-year = human-friendly explanation  