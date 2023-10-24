
# %%
import mlflow
logged_model = 'models:/NER_HF/4'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# %%
# %%
# Predict on a Pandas DataFrame.
import pandas as pd
df = pd.DataFrame([
    {'text': 'Hola que tal, soy Julian.'}
])
# %%
loaded_model.predict(df)
# %%












De 120 moderados había 50% de corruptos
Se moderaron 2000 y quedaron 244 más 
De los 364 hay 78 corruptos en total (21%)


Recall Yolo sino esta corrupto 100%
Recall de la correlación a nivel de marca da 1 (umbral: frena con 0.98, 0.9)
