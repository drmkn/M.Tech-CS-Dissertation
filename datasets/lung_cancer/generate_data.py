from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from utils import PREFIX
# Define the structure (DAG) based on the given BIF
model = DiscreteBayesianNetwork([
    ('asia', 'tub'),
    ('smoke', 'lung'),
    ('smoke', 'bronc'),
    ('lung', 'either'),
    ('tub', 'either'),
    ('either', 'xray'),
    ('bronc', 'dysp'),
    ('either', 'dysp')
])

# Define the CPDs
cpd_asia = TabularCPD('asia', 2, [[0.01], [0.99]])
cpd_smoke = TabularCPD('smoke', 2, [[0.5], [0.5]])

cpd_tub = TabularCPD('tub', 2,
                     [[0.05, 0.01],  # P(tub=yes | asia)
                      [0.95, 0.99]],  # P(tub=no | asia)
                     evidence=['asia'], evidence_card=[2])

cpd_lung = TabularCPD('lung', 2,
                      [[0.1, 0.01],   # P(lung=yes | smoke)
                       [0.9, 0.99]],  # P(lung=no | smoke)
                      evidence=['smoke'], evidence_card=[2])

cpd_bronc = TabularCPD('bronc', 2,
                       [[0.6, 0.3],   # P(bronc=yes | smoke)
                        [0.4, 0.7]],  # P(bronc=no | smoke)
                       evidence=['smoke'], evidence_card=[2])

cpd_either = TabularCPD('either', 2,
                        [[1, 1, 1, 0],  # P(either=yes | lung, tub)
                         [0, 0, 0, 1]],  # P(either=no | lung, tub)
                        evidence=['lung', 'tub'], evidence_card=[2, 2])

cpd_xray = TabularCPD('xray', 2,
                      [[0.98, 0.05],   # P(xray=yes | either)
                       [0.02, 0.95]],  # P(xray=no | either)
                      evidence=['either'], evidence_card=[2])

cpd_dysp = TabularCPD('dysp', 2,
                      [[0.9, 0.7, 0.8, 0.1],   # P(dysp=yes | bronc, either)
                       [0.1, 0.3, 0.2, 0.9]],  # P(dysp=no  | bronc, either)
                      evidence=['bronc', 'either'], evidence_card=[2, 2])

# Add CPDs to model
model.add_cpds(cpd_asia, cpd_smoke, cpd_tub, cpd_lung, cpd_bronc, cpd_either, cpd_xray, cpd_dysp)

# Validate the model
assert model.check_model()

# Sample data
sampler = BayesianModelSampling(model)
df_asia = sampler.forward_sample(size=3000, seed=42)

# Save to CSV
csv_file = PREFIX + "datasets/lung_cancer/data.csv"
df_asia.to_csv(csv_file, index=False)
