from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import TabularCPD

from pgmpy.inference import VariableElimination



# Create a Bayesian Network model
model = BayesianNetwork([('Diff', 'grades'), ('Intel', 'grades')])


# Define the CPDs (Conditional Probability Distributions)
cpd_Diff = TabularCPD(variable='Diff', variable_card=2, values=[[0.5], [0.5]])  # 70% chance of no rain, 30% chance of rain
cpd_Intel = TabularCPD(variable='Intel', variable_card=2, values=[[0.5], [0.5]]) 
cpd_traffic = TabularCPD(variable='grades', variable_card=2, values=[[0.6, 0.9, 0.1, 0.7], [0.4, 0.1, 0.9, 0.3]],
                         evidence=['Intel','Diff'], evidence_card=[2,2])  # Traffic depends on Rain

model.add_cpds(cpd_Diff, cpd_Intel, cpd_traffic)
# Perform inference using Variable Elimination
inference = VariableElimination(model)

# Query the probability of "Accident" given the evidence "Rain = 1" (it is raining)
evidence = {'grades': 1}
result = inference.query(variables=['Intel'], evidence=evidence)
print(result)



