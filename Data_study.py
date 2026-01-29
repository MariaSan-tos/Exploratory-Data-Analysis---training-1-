import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

csv = pd.read_csv('/content/BC250_2017_Terra_Indigena_A.csv') #got it from KAGGLE!

csv['areaoficial_km2'] = csv['areaoficialha'] / 100

plt.figure(figsize=(10,6))
sns.histplot(csv['areaoficial_km2'], bins=50, kde=True)
plt.title('Distribuição das Áreas das Terras Indígenas (km²)')
plt.xlabel('Área (km²)')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8,6))
sns.countplot(y='situacaojuridica', data=csv, order=csv['situacaojuridica'].value_counts().index)
plt.title('Distribuição por Situação Jurídica')
plt.xlabel('Número de Terras')
plt.ylabel('Situação Jurídica')
plt.show()

media_area = csv['areaoficial_km2'].mean()
mediana_area = csv['areaoficial_km2'].median()
desvio_padrao_area = csv['areaoficial_km2'].std()
area_min = csv['areaoficial_km2'].min()
area_max = csv['areaoficial_km2'].max()

print(f"Média da Área: {media_area} km²")
print(f"Mediana da Área: {mediana_area} km²")
print(f"Desvio Padrão da Área: {desvio_padrao_area} km²")
print(f"Área Mínima: {area_min} km²")
print(f"Área Máxima: {area_max} km²")


