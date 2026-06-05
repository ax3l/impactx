"""Physical constants (SI), matching ablastr::constant::SI used by ImpactX."""

#: speed of light in vacuum, m/s
c = 299792458.0
#: elementary charge, C
q_e = 1.602176634e-19
#: electron mass, kg
m_e = 9.1093837015e-31
#: proton mass, kg
m_p = 1.67262192369e-27

#: 1 MeV/c^2 expressed in kg  (== 1e6 * q_e / c^2)
MeV_inv_c2 = 1.0e6 * q_e / (c * c)

#: electron/positron anomalous magnetic moment (g-2)/2 [unitless]
g_anomaly_electron = 0.00115965218062
#: proton anomalous magnetic moment [unitless]
g_anomaly_proton = 1.7928473446
