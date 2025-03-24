#!/usr/bin/env python
# coding: utf-8

# # Ice, campañas publicitarias 2017

# ## Descripción del proyecto e introducción.

# La tienda online Ice que vende videojuegos por todo el mundo tiene las reseñas de usuarios y expertos, los géneros, las plataformas (por ejemplo, Xbox o PlayStation) y los datos históricos sobre las ventas de juegos que están disponibles en fuentes abiertas. El proposito del proyecto es identificar patrones que determinen si un juego tiene éxito o no. Esto permitirá detectar proyectos prometedores y planificar campañas publicitarias.
# 
# Hay datos que se remontan a 2016 y se está planeando una campaña para 2017.
# 
# El dataset contiene una columna "rating" que almacena la clasificación ESRB de cada juego. El Entertainment Software Rating Board (la Junta de clasificación de software de entretenimiento) evalúa el contenido de un juego y asigna una clasificación de edad como Adolescente o Adulto.

# ### Descripción de los datos.

# — Name (Nombre)
# 
# — Platform (Plataforma)
# 
# — Year_of_Release (Año de lanzamiento)
# 
# — Genre (Género) 
# 
# — NA_sales (ventas en Norteamérica en millones de dólares estadounidenses) 
# 
# — EU_sales (ventas en Europa en millones de dólares estadounidenses) 
# 
# — JP_sales (ventas en Japón en millones de dólares estadounidenses) 
# 
# — Other_sales (ventas en otros países en millones de dólares estadounidenses) 
# 
# — Critic_Score (máximo de 100) 
# 
# — User_Score (máximo de 10) 
# 
# — Rating (ESRB)
# 
# Es posible que los datos de 2016 estén incompletos.

# ## Presentación de los datos.

# In[88]:


# Importar los paquetes para el análisis.
import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st


# In[2]:


# Extraer el dataframe de los datos.
games = pd.read_csv('/datasets/games.csv')
games.head()


# ### Estadisticas descriptivas de los datos.

# Observaremos el estado original de los datos, si existen valores ausentes, el typo de almacenamiento de cada columna y algunas primeras estádisticas descriptivas para darnos una idea de nuestro dataset.
# 
# Iniciaremos detallando la información de las columnas como valores ausentes y el tipo de datos de cada una.

# In[3]:


# Extraer información de las columnas.
games.info()


# Tenemos un total de 16715 entradas o filas, en donde muchas columnas presentan datos ausentes, analizaremos esto con más detalle más adelante.
# 
# Al parecer solo hay dos columnas con el tipo de dato erroneo y se trata de Year_of_Release que trataremos a esta columna para convertirla a un entero y de User_Score que convertiremos a numero flotante.

# In[4]:


# Estadisticas descriptivas de las columnas numericas.
games.describe()


# Se aprecia que el menor año observable en los datos es 1980 y el más reciente 2016 y que algunos juegos no tienen años registrados. Tambien que más de la mitad de los juegos no tienen un Critic_Score.
# 
# En las ventas observando la distribución de los datos nos damos cuenta que al menos el 25% de los datos de cada columna tiene 0 como valores de venta, este dato puede ser interesante para indagar más a fondo en el.

# In[5]:


# Estadisticas descriptivas de las columnas categoricas.
games.describe(include=['object'])


# Existen dos juegos con la columna Name nula y con la columna Genre nula, deberíamos echar un vistazo a sus datos.
# 
# Entre otras abstracciones conocemos que hay 12 tipos diferentes de géneros, 8 tipos de Ratings y 31 tipos de plataformas.
# 
# También que la columna User_Score no solo tiene 2424 juegos tbd (a ser determinados), si no que también presenta una gran cantidad de valores ausentes como también pasa con Rating.

# ## Limpieza de datos

# ### Limpieza de encabezados

# Nos encargaremos de unificar el estilo de los encabezados para facilitar su uso haciendo que todas las letras esten en minusculas.

# In[6]:


# Reemplazar los nombres de las columnas a minusculas.
new_columns = []
for column in games.columns:
    new_columns.append(column.lower())

# Asignar los nuevos nombres a las columnas.
games.columns = new_columns
games.columns


# In[7]:


games.head() 


# ### Reasignación del tipo de dato.

# Como ya mencionamos convertiremos el tipo de dato de las columnas year_of_release y user_score al tipo correcto.
# 
# year_of_release requiere un cambio al tipo entero ya que los años no suelen expresar fracciones si no valores completos, igualmente user_score podría requerir aplicacion de operaciones aritmeticas y siento tipo object no será viable, esta es la razon por la que debe ser transformado a tipo float.

# In[8]:


# Reasignando el tipo de year_of_release
games['year_of_release'] = games['year_of_release'].astype('Int32')
games['year_of_release']


# En el caso de user_score debemos tener en cuenta que el registra algunos valores ausentes como 'tbd' sin embargo estos no aportan nada de información por lo que los trataremos igual que un valor nulo NaN, para ello debemos hacer un paso antes de la transformación del tipo a float.

# In[9]:


# Reasignando el tipo de user_score
# Transformando valores 'tbd' a NaN
games.loc[games['user_score'] == 'tbd', 'user_score'] = np.nan

# Asignando el tipo a float
games['user_score'] = games['user_score'].astype('float')
games['user_score']


# ### Tratamiento de valores ausentes.

# Para iniciar este proceso, observemos nuevamente las columnas que presentan valores nulos y la cantidad de filas nulas que exponen.

# In[10]:


# Revision de columnas con valores nulos y su cantidad.
games.isna().sum()[games.isna().sum() > 0]


# #### Columnas name y genre.

# Comenzemos estudiando los valores nulos de name y genre los cuales aparecen en las mismas filas y bajo algunas coincidencias interesantes.

# In[11]:


# Filas con valores nulos de name
games[games['name'].isna()]


# En este par de filas podemos observar algunas situaciones relevantes, como por ejemplo que la plataforma y el año son compartidos y que el valor de género tambien es nulo en las dos filas.
# 
# Realmente de estas filas las unicas columnas con información relevante son las ventas por país es por eso que intentaremos reunir toda su informacion util en una sola fila que tendra como nombre 'Unknown' y como genero 'Unknown' támbien, de esta manera no descartaremos sus valores de ventas por país pero tampoco asignaremos estos valores a otra categoria o genero a la que tal vez no corresponda alterando posibles conclusiones y analisis.

# In[12]:


# Reasignaremos todos los datos utiles a la fila 659.
games.loc[659, 'name'] = 'Unknown'
games.loc[659, 'genre'] = 'Unknown'
games.loc[659, 'jp_sales'] = 0.03

# Descartamos la segunda fila con indice 14244.
games.dropna(subset=['name'], inplace=True)
games.reset_index(drop=True, inplace=True)

games[games['name'] == 'Unknown']


# In[13]:


# Verificación de columnas con valores nulos y su cantidad.
games.isna().sum()[games.isna().sum() > 0]


# Ahora toda la información relevante se encuentra en un solo juego 'Unknown' rescatando los valores de ventas por país y no alterando los valores de ventas por año ni por genero registradas.

# #### Columna year_of_release.

# In[14]:


# Filas con valores nulos de year_of_release.
games[games['year_of_release'].isna()]


# Para esta columna se observa que tenemos 269 valores nulos, para poder asignarles un valor se pensaron en las siguientes alternativas:
# 
# - Utilizar los años de los mismos juegos (nombres) que se lanzaron en otras plataformas.
# - Utilizar la información que se pudiera encontrar en el nombre del juego.
# 
# A pesar de ser dos opciones muy interesantes no podemos inclinarnos por ninguna de las dos ya que en muchos casos los juegos son lanzados para diferentes consolas en años distintos y puede cambiar por región, por lo que la primera opción se descartaría. También el hecho de que en el titulo del juego se incluya el año (como sucede en muchos juegos deportivos como FIFA Soccer 2004) no nos asegura que el año de lanzamiento halla sido ese ya que hay casos en que si coincide y hay otros en donde es el año anterior.
# 
# No teniendo acceso a otras tablas o información que nos den un mayor respaldo de estos datos se opto por dejar estos años como valores nulos y descartar estas filas en analisís en donde se requiera de un filtro por año.

# #### Columnas critic_score y user_score.

# In[15]:


# Filas con valores nulos de critic_score.
games[games['critic_score'].isna()]


# In[16]:


# Filas con valores nulos de user_score.
games[games['user_score'].isna()]


# Las columnas critic_score y user_score contienen información muy relevante para el analisís de datos que realizaremos por lo que una incorrecta asignación de valores ausentes alterará por completo los resultados y podría terminar en conclusiones incorrectas, esto teniendo en cuenta que la cantidad de valores ausentes en estas dos columnas ronda el 50% del total de los datos lo cual represanta una porción muy importante del dataset, debemos reflexionar muy bien en esta etapa.
# 
# Al observar las calificaciones de los mismos titulos pero en plataformas distintas nos damos cuenta que estas difieren por lo que no podemos extraer datos de estas coincidencias para ninguna de las dos columnas.
# 
# Teniendo en cuenta lo anterior, lo más prudente sería no alterar estos datos y mantener los valores nulos como tal mientras no tengamos otra fuente de información fiable y complementaria, alterar o llenar los valores nulos con la media o la mediana implicaria cambiar alrededor del 50% de datos y es una porcion muy significativa que podría modificar resultados evaluativos que involucren estas columnas en conjunto con otras.

# In[17]:


# Juegos que comparten nombre pero en diferentes plataformas.
games[games.duplicated(subset=['name'], keep=False)].sort_values('name')


# Aquí confirmamos que a pesar de que los titulos compartan nombre difieren en su calificación dependiendo de su plataforma.

# La razón de esta cantidad de valores ausentes esta relacionada tanto con el año de lanzamiento del juego como con su plataforma, a continuacion podemos ver un par de tablas que nos demuestran que las reseñas son casi nulas hasta después de 1995 y que existen multiples plataformas en donde las reseñas no son tenidas en cuenta.

# In[18]:


# Cantidad de reseñas por año de lanzamiento.
games.groupby('year_of_release')['year_of_release', 'critic_score', 'user_score'].count()


# In[19]:


# Cantidad de reseñas por plataforma.
games.groupby('platform')['platform', 'critic_score', 'user_score'].count()


# #### Columna rating.

# In[20]:


# Filas con valores nulos de rating.
games[games['rating'].isna()]


# In[21]:


# Explorando si un mismo juego puede tener mas de un rating.
# Se hallan juegos con el mismo nombre, se agrupan luego por nombre y en su columna rating se identifica cuantos valores registrados unicos existen.
games[(games.duplicated(subset=['name'], keep=False))].groupby('name')['rating'].agg(['unique', 'nunique']).sort_values('nunique', ascending=False)


# La columna rating es algo similar a las dos columnas que recien estudiamos, ya que la cantidad de valores nulos ronda el 40% del dataframe, a esta columna tampoco podemos reasignarle los valores nulos basandonos en los mismos titulos de otras plataformas ya que evidenciamos que estas pueden variar entre plataformas a pesar de ser el mismo titulo. También para años mayores a 1995 se inicia a tener realmente una cantidad considerable de registros de este dato y de la misma manera algunas consolas no registran información al respecto.

# In[22]:


# Cantidad de filas con rating por año de lanzamiento.
games.groupby('year_of_release')['year_of_release', 'rating'].count()


# In[23]:


# Cantidad de filas con rating por plataforma.
games.groupby('platform')['platform', 'rating'].count()


# Dado lo visto y sin información adicional la columna rating permanecera con sus valores nulos intactos.

# ### Tratamiento de valores duplicados.

# Revisaremos valores duplicados para 3 tipos de coincidencias distintas entre las columnas name, platform y year_of_release:

# In[24]:


# Valores duplicados para las tres columnas.
games[games.duplicated(['name', 'platform', 'year_of_release'], keep=False)]


# Al parecer solo existe una coincidencia exacta para las 3 columnas y al parecer la información más completa la maneja la primera fila, por lo que será la fila que mantengamos en el dataframe.

# In[25]:


# Mantenemos la primera coincidencia y descartamos la segunda.
games.drop_duplicates(subset=['name', 'platform', 'year_of_release'], keep='first', inplace=True, ignore_index=True)


# In[26]:


# Evaluamos valores duplicados en nombre y plataforma.
games[games.duplicated(['name', 'platform'], keep=False)]


# En este caso a pesar de observar más valores duplicados el hecho de que el año de lanzamiento sea diferente nos brinda información relevante ya que muchos juegos son lanzados nuevamente en las mismas plataformas pero de forma actualizada (remakes), es por esto que estos valores no serán descartados ya que se tratan de juegos y entregas distintas.

# ### Creación de columnas de valor.

# Calcularemos las ventas totales (la suma de las ventas en todas las regiones) para cada juego y añadiremos estos valores en una columna separada.

# In[27]:


# Nueva columna de ventas totales
games['total_sales'] = games['na_sales'] + games['eu_sales'] + games['jp_sales'] + games['other_sales']
games


# ## Análisis de los datos.

# ### Analisis exploratorio.

# Iniciaremos nuestro analisis observando los lanzamientos que se realizaron en cada año y las ganancias generadas por estos juegos.

# In[28]:


# Juegos lanzados por año junto con el total de ventas anuales.
year_evaluation = games.groupby('year_of_release')['total_sales'].agg(['count', 'sum'])
year_evaluation.columns = ['games_released', 'total_sales']
year_evaluation


# In[29]:


# Vizualizando los datos de juegos por año.
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=year_evaluation, x=year_evaluation.index, y='games_released')

#Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Juegos lanzados por año')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Numero de juegos')
plt.xticks(rotation=45)
plt.show()


# In[30]:


# Vizualizando los ingresos de juegos por año.
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=year_evaluation, x=year_evaluation.index, y='total_sales')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Ingresos totales por año')
plt.xlabel('Año de lanzamiento')
plt.ylabel('Ventas totales en millones (USD)')
plt.xticks(rotation=45)
plt.show()


# Podemos destacar que a partir del año **1994** la cantidad de juegos lanzados incrementa considerablemente y que a partir del año **1996** los ingresos anuales tambien se incrementan.

# Ahora estudiemos como varían las ventas de una plataforma a otra en el tiempo.

# In[31]:


# Juegos lanzados por plataforma junto con el total de ventas anuales.
platform_evaluation = games.groupby(['platform'])['total_sales'].agg(['count', 'sum'])
platform_evaluation.columns = year_evaluation.columns = ['games_released', 'total_sales']
platform_evaluation


# In[32]:


# Organizando por numero de juegos de mayor a menor
platform_evaluation.sort_values('games_released', ascending=False, inplace=True)

# Vizualizando los juegos por plataforma.
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=platform_evaluation, x=platform_evaluation.index, y='games_released')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Juegos lanzados por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Número de juegos')
plt.xticks(rotation=45)
plt.show()


# Podemos observar que hay un gran contraste entre algunas plataformas y la cantidad de juegos lanzados en ellas, esto lo tendremos en cuenta junto con los siguientes analisis para saber que datos son los más relevantes para nuestros estudios.

# In[33]:


# Organizando por ventas totales de mayor a menor
platform_evaluation.sort_values('total_sales', ascending=False, inplace=True)

# Vizualizando los ingresos totales por plataforma.
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=platform_evaluation, x=platform_evaluation.index, y='total_sales')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Ingresos totales por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas totales en millones (USD)')
plt.xticks(rotation=45)
plt.show()


# En la distribución de ganancias por plataforma podemos segregar las plataformas en 2 categorias: Altas (mayores a 600 millones de dolares), Bajas (menores a 600 millones de dolares). Estudiemos la distribución de las ganancias altas a través del tiempo.

# In[34]:


# Encontrando las plataformas con ventas altas
high_condition = platform_evaluation['total_sales'] >= 400
platforms_high = list(platform_evaluation[high_condition].index)

platforms_high


# In[35]:


# Filtrando las tablas de datos para las plataformas de ventas altas.
high_platform_evaluation_yearly = games[games['platform'].isin(platforms_high)]

# Encontrando el numero de juegos y las ventas por año y por plataforma.
high_platform_evaluation_yearly = high_platform_evaluation_yearly.groupby(['platform', 'year_of_release'])['total_sales'].agg(['count', 'sum'])
high_platform_evaluation_yearly.columns = high_platform_evaluation_yearly.columns = ['games_released', 'total_sales']
high_platform_evaluation_yearly.reset_index(inplace=True)

high_platform_evaluation_yearly


# In[36]:


# Visualizando los resultados.
plt.figure(figsize=(10, 6))

ax = sns.lineplot(data=high_platform_evaluation_yearly, x='year_of_release', y='total_sales', hue='platform')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Plataformas de altos ingresos a través del tiempo')
plt.xlabel('Año')
plt.ylabel('Ventas totales en millones (USD)')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# Obserbando los valores de las ganancias a través del tiempo de las plataformas con mayores ingresos notamos que son plataformas que lanzan videojuegos desde el año **1994**, todas manifiestan un pico de ventas unos años despues de su lanzamiento y luego un descenso.
# 
# __Nota: el primer dato de la plataforma DS es un dato erroneo, ya que esta plataforma fue lanzada en el 2004.__

# Estudiemos otro sesgo, revisemos las plataformas basandonos en el ultimo año que fue registrado un videjuego para su plataforma, seran antiguas si su juego mas reciente fue lanzado antes del año 2000, intermedias si estan entre el 2000 y el 2010 y recientes para años mayores o iguales al 2010.

# In[37]:


# Identificando el ultimo año que cada plataforma lanzo un juego.
max_dates = games.groupby('platform')['year_of_release'].max().sort_values()
max_dates


# In[38]:


# Encontrando las plataformas mas antiguas.
old_platforms = list(max_dates[max_dates < 2000].index)

old_platforms


# In[39]:


# Filtrando las tablas de datos para las plataformas antiguas.
old_platform_evaluation_yearly = games[games['platform'].isin(old_platforms)]

# Encontrando el numero de juegos y las ventas por año y por plataforma.
old_platform_evaluation_yearly = old_platform_evaluation_yearly.groupby(['platform', 'year_of_release'])['total_sales'].agg(['count', 'sum'])
old_platform_evaluation_yearly.columns = old_platform_evaluation_yearly.columns = ['games_released', 'total_sales']
old_platform_evaluation_yearly.reset_index(inplace=True)

old_platform_evaluation_yearly.head()


# In[40]:


# Visualizando los resultados.
plt.figure(figsize=(10, 6))

ax = sns.lineplot(data=old_platform_evaluation_yearly, x='year_of_release', y='total_sales', hue='platform')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Plataformas antiguas a través del tiempo')
plt.xlabel('Año')
plt.ylabel('Ventas totales en millones (USD)')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# Al observar la gráfica de plataformas antiguas a través del tiempo, notamos que ninguna de ellas supera los 60 millones de dolares en ningun año, tendremos esto en cuenta para futuros análisis.

# In[41]:


# Encontrando las plataformas intermedias.
mid_platforms = list(max_dates[(max_dates >= 2000) & (max_dates < 2010)].index)

mid_platforms


# In[42]:


# Filtrando las tablas de datos para las plataformas intermedias.
mid_platform_evaluation_yearly = games[games['platform'].isin(mid_platforms)]

# Encontrando el numero de juegos y las ventas por año y por plataforma.
mid_platform_evaluation_yearly = mid_platform_evaluation_yearly.groupby(['platform', 'year_of_release'])['total_sales'].agg(['count', 'sum'])
mid_platform_evaluation_yearly.columns = mid_platform_evaluation_yearly.columns = ['games_released', 'total_sales']
mid_platform_evaluation_yearly.reset_index(inplace=True)

mid_platform_evaluation_yearly


# In[43]:


# Visualizando los resultados.
plt.figure(figsize=(10, 6))

ax = sns.lineplot(data=mid_platform_evaluation_yearly, x='year_of_release', y='total_sales', hue='platform')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Plataformas intermedias a través del tiempo')
plt.xlabel('Año')
plt.ylabel('Ventas totales en millones (USD)')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# Este grupo de plataformas alcanzan mejores ventas en su mayoría, si la comparamos con el grupo previo. Destacamos que la plataforma PS es la unica del grupo de 'altas ventas' que pertenece tambien a este grupo haciendo gran contraste con las demás plataformas.
# 
# Tambien notamos que todas las plataformas a excepcion de GB tienen su primer registro a partir de **1994**.

# In[44]:


# Encontrando las plataformas recientes.
recent_platforms = list(max_dates[max_dates >= 2010].index)

recent_platforms


# In[45]:


# Filtrando las tablas de datos para las plataformas recientes.
new_platform_evaluation_yearly = games[games['platform'].isin(recent_platforms)]

# Encontrando el numero de juegos y las ventas por año y por plataforma.
new_platform_evaluation_yearly = new_platform_evaluation_yearly.groupby(['platform', 'year_of_release'])['total_sales'].agg(['count', 'sum'])
new_platform_evaluation_yearly.columns = new_platform_evaluation_yearly.columns = ['games_released', 'total_sales']
new_platform_evaluation_yearly.reset_index(inplace=True)

new_platform_evaluation_yearly


# In[46]:


# Visualizando los resultados.
plt.figure(figsize=(10, 6))

ax = sns.lineplot(data=new_platform_evaluation_yearly, x='year_of_release', y='total_sales', hue='platform')

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_xticks(year_evaluation.index.astype(int))
plt.title('Plataformas recientes a través del tiempo')
plt.xlabel('Año')
plt.ylabel('Ventas totales en millones (USD)')
plt.legend(title='Plataforma')
plt.xticks(rotation=45)
plt.show()


# En la grafica de plataformas recientes a través del tiempo podemos observar una mayor cantidad de ingresos que en el resto de grupos, sin embargo no todas las plataformas cumplen esta premisa, ya que plataformas como XOne, PC o WiiU no muestran grandes ingresos.
# 
# Otra conclusion interesante es que la plataforma con mayor permanencia en el mercado es el PC.
# 
# __Nota: el primer dato de la plataforma DS es un dato erroneo, ya que esta plataforma fue lanzada en el 2004.__

# Para continuar, revisaremos la permanencia de las plataformas en el mercado, identificando el primer año en el que un juego es lanzado en la plataforma y el ultimo.

# In[47]:


def categorize_year(year):
    if year < 2000:
        return 'old'
    elif year < 2010:
        return 'mid'
    else:
        return 'new'


# In[48]:


# Identificando el primer año que cada plataforma lanzo un juego.
min_dates = games.groupby('platform')['year_of_release'].min().sort_values()

# Unimos las dos columnas y encontramos la diferencia entre las dos.
games_duration = pd.concat([min_dates, max_dates], axis=1)
games_duration.columns = ['first_year', 'last_year']

games_duration.loc['DS', 'first_year'] = 2004 # Realizamos la correcion mencionada anteriormente para DS.

games_duration.reset_index(inplace=True)
games_duration['permanence'] = games_duration['last_year'] - games_duration['first_year']
games_duration.sort_values('permanence', ascending=False, ignore_index=True, inplace=True)

# Aplicar la función a la columna 'last_year'
games_duration['group'] = games_duration['last_year'].apply(categorize_year)

# Organizar por grupo y permanencia
group_order = ['old', 'mid', 'new']
games_duration['group'] = pd.Categorical(games_duration['group'], categories=group_order, ordered=True)
games_duration.sort_values(['group', 'permanence'], ascending=[False, False], ignore_index=True, inplace=True)
games_duration


# In[49]:


# Visualizando los resultados.
plt.figure(figsize=(10, 6))
sns.barplot(data=games_duration, x='platform', y='permanence', hue='group', hue_order=group_order)

# Dando formato al grafico
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Permanencia de las plataformas en el mercado')
plt.xlabel('Plataforma')
plt.ylabel('Años en el mercado')
plt.legend(title='Grupo')
plt.xticks(rotation=45)
plt.show()


# In[50]:


# Promedio de duracion en el mercado de las plataformas por grupo.
games_duration.groupby('group')['permanence'].mean()


# Podemos apreciar como a través del tiempo las plataformas permanecen más en los años, las plataformas mas antiguas tienen una duración promedio de 3.9 años mientras que las más recientes permanecen en promedio 9.4 años, esto puede estar relacionado con que muchas de las plataformas recientes recauden una mayor cantidad de dinero.

# ### Filtrado de datos para el modelo 2017.

# Determinando nuestros resultados aplicaremos un filtro por año en donde solo tendremos en cuenta los juegos desde el año 1994 en adelante, aquí estan las razones por las que decidimos que este filtro podría ser representativo para nuestro estudio:
# 
# - A partir de 1996 se tienen registros representativos para las valoraciones critic_score, user_score y rating, sin embargo a partir de un par de años atras, es decir 1994, se inicia a notar un aumento en el lanzamiento de videojuegos que aumenta considerablemente año tras año.
# - El primer registro de las plataformas con altos ingresos se tiene en 1994, y lo consigue la plataforma PS.
# - Las plataformas recientes e intermedias reparten sus ganancias en gran mayoria para años posteriores a 1994, siendo las unicas excepciones las plataformas PC y GB, sin embargo hay suficientes valores que las representen con ventas posteriores a 1994.

# In[51]:


# Filtrando los juegos con años de lanzamiento mayores e iguales a 1994.
games_filtered = games[games['year_of_release'] >= 1994].reset_index(drop=True)
games_filtered


# Tambien se descartaran los datos de las plataformas que no superen en ventas totales los 10 millones de dolares, ya que no contienen datos de impacto para el estudio.

# In[52]:


# Identificando las plataformas que seran descartadas por los criterios mencionados.
platforms_total_sales = games_filtered.groupby('platform')['total_sales'].sum().sort_values()

# Describing conditions
platforms_to_exclude = list(platforms_total_sales[platforms_total_sales < 10].index)

platforms_to_exclude


# In[53]:


# Filtrando los juegos por la nueva condicion de plataformas.
games_filtered = games_filtered[~games_filtered['platform'].isin(platforms_to_exclude)]
games_filtered.reset_index(drop=True, inplace=True)
games_filtered


# ### Análisis exploratorio del nuevo dataset.

# Ahora tenemos datos más representativos para nuestro estudio del 2017, recapitulemos algunas observaciones para contemplar el nuevo dataset, por ejemplo, observemos las distribuciones de las ventas totales por plataforma. 

# In[54]:


# Diagrama de caja para visualizar la distribución de las ventas globales de todos los juegos, desglosados por plataforma.
plt.figure(figsize=(10, 6))
sns.boxplot(data=games_filtered, x="total_sales", y="platform", palette="Set2")

# Agregar título y etiquetas a los ejes
plt.title("Distribución de ventas totales por plataforma", fontsize=14)
plt.xlabel("Ventas en millones (USD)", fontsize=12)
plt.ylabel("Plataforma", fontsize=12)

# Mostrar el gráfico
plt.show()


# Al observar las distribuciones de ventas totales por plataforma podemos ver una gran cantidad de valores atipicos que sesgan hacia la derecha en todas las distribuciones, esto significa que existen valores extremos que afectan el promedio fuertemente por lo que el uso de la mediana es una medida de tendencia central mas apropiada para nuestro dataset.

# Analicemos ahora las plataformas líderes en ventas que más crecen basandonos en la media y mediana de las ventas totales y en si es una plataforma reciente, a estas las llamaremos plataformas potencialmente rentables.

# In[55]:


# Agrupando los datos por numero de juegos y ventas
most_valuable_platforms = games_filtered.groupby(['platform'])['total_sales'].agg(['count', 'sum', 'mean','median', 'std'])

# Organizando los datos
most_valuable_platforms.columns = ['games_released', 'total_sales', 'mean_sales', 'median_sales', 'std_sales']
most_valuable_platforms.sort_values(['median_sales'], ascending=False, inplace=True)
most_valuable_platforms.reset_index(inplace=True)

# Se filtran las plataformas recientes.
most_valuable_platforms = most_valuable_platforms[most_valuable_platforms['platform'].isin(recent_platforms)]
most_valuable_platforms.reset_index(drop=True, inplace=True)
most_valuable_platforms


# In[56]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=most_valuable_platforms, x='platform', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=most_valuable_platforms, x='platform', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(most_valuable_platforms['median_sales'].max(), most_valuable_platforms['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Plataformas potencialmente rentables')
ax1.set_xlabel('Plataforma')

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Podemos concluir que las 5 plataformas potencialmente más rentables son: X360, PS3, PS4, Wii y XOne. Esto debido a que tienen los mayores promedios y medianas en ventas totales y tienen la oportunidad de crecimiento más alta al ser plataformas recientes.
# 
# A pesar de que Ps2 y WiiU tienen mejores medianas, las medias de XOnes, PS4 y Wii son considerablemente más altas y presentan medianas muy cercanas a estas dos primeras plataformas.
# 
# Es bueno notar, que Wii es la plataforma con la mayor desviacion estandar (superior a 3), muy por encima del resto de plataformas las cuales se encuentran en valores al rededor de 1.
# 
# Podríamos estudiar como afectan otras variables la distribucion de ventas totales de la esta plataforma (Wii) y si estan relacionadas con este nivel de dispersión más elevado, por ejemplo las reseñas profesionales y de usuarios.

# In[57]:


# Seleccionando los juegos de Wii.
wii_games = games_filtered[games_filtered['platform'] == 'Wii']
wii_games


# In[58]:


# Detectando la correlacion entre ventas totales, reseñas profesionales y reseñas de usuarios.
corr_matrix = wii_games[['total_sales', 'critic_score', 'user_score']].corr()

# Crear el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Configuración del título
plt.title('Mapa de Correlaciones - Juegos de Wii')
plt.show()


# Al parecer ni las reseñas por profesionales ni las reseñas por usuarios estan correlacionas a las ventas totales, los valores de 0.18 y 0.11 respectivamente lo comprueban. La unica correlación que parece existir es entre las reseñas de profesionales y usuarios, siendo esta una correlación positiva de 0.69, esto significaría que profesionales y usuarios pueden persivir un juego de maneras similares y calificarlo de forma similar.
# 
# Exploremos estas correlaciones tambien en las plataformas X360 y PS4, quienes fueron las plataformas con mayor mediana y media respectivamente.

# In[59]:


# Seleccionando los juegos de X360.
x360_games = games_filtered[games_filtered['platform'] == 'X360']

# Detectando la correlacion entre ventas totales, reseñas profesionales y reseñas de usuarios.
corr_matrix = x360_games[['total_sales', 'critic_score', 'user_score']].corr()

# Crear el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Configuración del título
plt.title('Mapa de Correlaciones - Juegos de X360')
plt.show()


# El resultado es muy similar en X360 comparado con Wii, la única diferencia apreciable es que la correlación de reseñas profesionales aumenta ligeramente pero no lo suficiente para mantener una correlación fuerte o media, continua siendo debil.

# In[60]:


# Seleccionando los juegos de PS4.
ps4_games = games_filtered[games_filtered['platform'] == 'PS4']

# Detectando la correlacion entre ventas totales, reseñas profesionales y reseñas de usuarios.
corr_matrix = ps4_games[['total_sales', 'critic_score', 'user_score']].corr()

# Crear el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Configuración del título
plt.title('Mapa de Correlaciones - Juegos de PS4')
plt.show()


# Nuevamente el resultadoen PS4 refleja las mismas conclusiones,una correlación de reseñas profesionales que aumenta ligeramente pero que continua siendo debil. Aquí inclusive la correlación de reseñas con usuarios es cercana a 0.

# Para el próximo análisis tengamos en cuenta las plataformas mas rentables y como se diferencian en ventas para los mismos videojuegos.

# In[61]:


# Enlistando los juegos de las plataformas mas rentables.
MVP = ['X360', 'PS3', 'PS4', 'Wii', 'XOne']

# Seleccionando los juegos MVP
MVP_games = games_filtered[games_filtered['platform'].isin(MVP)]

# Agrupando por juegos y encontrando los juegos multiplataforma.
multiplatform = MVP_games.groupby('name')['platform'].count()
multiplatform = list(multiplatform[multiplatform> 1].index)

# Seleccionando los juegos multiplataforma MVP.
MVP_games_multiplatform = MVP_games[MVP_games['name'].isin(multiplatform)]
MVP_games_multiplatform


# In[62]:


# Seleccionando 3 juegos aleatorios.
games_to_evaluate = list(MVP_games_multiplatform['name'].sample(3, random_state=2)) # Cambiar la semilla (random_state) si se quieren analizar diferentes juegos.
games_to_evaluate


# In[63]:


# Comparacion de ventas totales juego 1.
game_1 = MVP_games_multiplatform[MVP_games_multiplatform['name'] == games_to_evaluate[0]]

#Visualizando comparación.
sns.barplot(data=game_1, x='platform', y='total_sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'Ventas de {games_to_evaluate[0]} por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas en millones de USD')

plt.show()


# In[64]:


# Comparacion de ventas totales juego 1.
game_2 = MVP_games_multiplatform[MVP_games_multiplatform['name'] == games_to_evaluate[1]]

#Visualizando comparación.
sns.barplot(data=game_2, x='platform', y='total_sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'Ventas de {games_to_evaluate[1]} por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas en millones de USD')

plt.show()


# In[65]:


# Comparacion de ventas totales juego 1.
game_3 = MVP_games_multiplatform[MVP_games_multiplatform['name'] == games_to_evaluate[2]]

#Visualizando comparación.
sns.barplot(data=game_3, x='platform', y='total_sales')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title(f'Ventas de {games_to_evaluate[2]} por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas en millones de USD')

plt.show()


# En los 3 casos podemos comprobar que las ventas pueden variar significativamente para un mismo juego en función de su plataforma. 

# Echemos un vistazo a la distribución general de los juegos por género. ¿Qué se puede decir de los géneros más rentables?

# In[66]:


# Diagrama de caja para visualizar la distribución de las ventas globales por genero.
plt.figure(figsize=(10, 6))
sns.boxplot(data=games_filtered, x='total_sales', y='genre', palette="Set2")

# Agregar título y etiquetas a los ejes
plt.title("Distribución de ventas totales por género", fontsize=14)
plt.xlabel("Ventas en millones (USD)", fontsize=12)
plt.ylabel("Género", fontsize=12)

# Mostrar el gráfico
plt.show()


# El comportamiento de la distribución por género es similar al comportamiento que observamos por plataforma, para cada género comportamientos con sesgos a la derecha con multiples valores atipicos. Se recomienda que se analicen los generos mas valiosos de forma similar.

# In[67]:


# Agrupando los datos por genero y ventas
most_valuable_genres = games_filtered.groupby('genre')['total_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
most_valuable_genres.columns = ['games_released', 'total_sales', 'mean_sales', 'median_sales', 'std_sales']
most_valuable_genres.sort_values(['median_sales'], ascending=False, inplace=True)
most_valuable_genres.reset_index(inplace=True)
most_valuable_genres


# In[68]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=most_valuable_genres, x='genre', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=most_valuable_genres, x='genre', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(most_valuable_genres['median_sales'].max(), most_valuable_genres['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Géneros potencialmente rentables')
ax1.set_xlabel('Género')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Los 5 géneros potencialmente más rentables son: Platform, Shooter, Sports, Role-Playing y Racing.
# 
# A pesar de que el género con más ventas totales es Action, los valores de media y mediana indican que esto puede deberse más a que existen una gran cantidad de titulos de este genero en vez de que las entregas de este género sean potencialmente rentables.
# 
# En el caso del género Fighting, los géneros Role-Playing y Racing demostraron un mejor promedio con tan solo 2 centecimas de diferencia en su mediana, esto los úbica de mejor manera como géneros potencialmente rentables.

# ### Análisis descriptivo por regiones.

# Ahora revisaremos las variables para cada una de las 3 regiones individualmente. Iniciemos por observar las ventas totales para cada región.

# In[69]:


# Extraer la sumatoria para cada venta de cada región.
na_sales = games_filtered['na_sales'].sum()
eu_sales = games_filtered['eu_sales'].sum()
jp_sales = games_filtered['jp_sales'].sum()
other_sales = games_filtered['other_sales'].sum()

# Generar un Dataframe para denerar un diagrama de barras.
sales = pd.DataFrame([{'sales':na_sales}, {'sales': eu_sales}, {'sales':jp_sales}, {'sales':other_sales}], index=['NA', 'EU', 'JP', 'Other'])

sales.plot(kind='bar', rot=0, legend=False, xlabel='Región', ylabel='Millones de dolares', title='Ventas totales por región')
plt.show()


# La región con mayores ventas es Norte America mientras que la región con menos ventas es Japón casi igualando las ventas en totales en otras partes de mundo. Para los siguientes análisis solo incluiremos las 3 regiones principales y excluiremos al resto del mundo.
# 
# Iniciemos estudiando el desempeño de cada plataforma por región.

# In[70]:


# Agrupando los datos por plataforma y region NA
na_platform_sales = games_filtered.groupby('platform')['na_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
na_platform_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
na_platform_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
na_platform_sales.reset_index(inplace=True)

# Se filtran las plataformas recientes.
na_platform_sales = na_platform_sales[na_platform_sales['platform'].isin(recent_platforms)]
na_platform_sales.reset_index(drop=True, inplace=True)
na_platform_sales


# In[71]:


# Agrupando los datos por plataforma y region EU
eu_platform_sales = games_filtered.groupby('platform')['eu_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
eu_platform_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
eu_platform_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
eu_platform_sales.reset_index(inplace=True)

# Se filtran las plataformas recientes.
eu_platform_sales = eu_platform_sales[eu_platform_sales['platform'].isin(recent_platforms)]
eu_platform_sales.reset_index(drop=True, inplace=True)
eu_platform_sales


# In[72]:


# Agrupando los datos por plataforma y region JA
jp_platform_sales = games_filtered.groupby('platform')['jp_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
jp_platform_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
jp_platform_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
jp_platform_sales.reset_index(inplace=True)

# Se filtran las plataformas recientes.
jp_platform_sales = jp_platform_sales[jp_platform_sales['platform'].isin(recent_platforms)]
jp_platform_sales.reset_index(drop=True, inplace=True)
jp_platform_sales


# In[73]:


# Comparando el desempeño de cada plataforma sesgado por región.
na_platform_sales['region'] = 'NA'
eu_platform_sales['region'] = 'EU'
jp_platform_sales['region'] = 'JP'

# Visualizacion de la información.
all_region_platform_sales = pd.concat([na_platform_sales[['platform', 'sales', 'region']], eu_platform_sales[['platform', 'sales', 'region']], jp_platform_sales[['platform', 'sales', 'region']]])

sns.barplot(data=all_region_platform_sales, x='platform', y='sales', hue='region', dodge=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Desempeño de cada región por plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Ventas en millones de USD')
plt.legend(title='Región')
plt.show()


# Ya que las ventas de Norte América son mayores estas destacan sobre las otras regiones, sin embargo las plataformas favoritas en Norte América son X360, PS2 y Wii. En el caso de Europa PS3, PS2 y X360 son quienes tienen más ventas. Finalmente Japón prefiere DS, PS2 y 3DS.
# 
# Al parecer Japon tiene ventas muy bajas en XOne y PC siendo estas las más bajas de este análisis.
# 
# Observemos el comportamiento de las ventas con más detalle en cada región.

# In[74]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=na_platform_sales, x='platform', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=na_platform_sales, x='platform', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(na_platform_sales['median_sales'].max(), na_platform_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Plataformas potencialmente rentables en Norte America')
ax1.set_xlabel('Plataforma')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Las 5 plataformas potencialmente más rentables en Norteamerica son X360, Wii, XOne, PS3 y PS2.

# In[75]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=eu_platform_sales, x='platform', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=eu_platform_sales, x='platform', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(eu_platform_sales['median_sales'].max(), eu_platform_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Plataformas potencialmente rentables en Europa')
ax1.set_xlabel('Plataforma')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Las 5 plataformas potencialmente más rentables en Europa son PS4, XOne, PS3, X360 y WiiU.

# In[76]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=jp_platform_sales, x='platform', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=jp_platform_sales, x='platform', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(jp_platform_sales['median_sales'].max(), jp_platform_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Plataformas potencialmente rentables en Japón')
ax1.set_xlabel('Plataforma')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Las 5 plataformas potencialmente más rentables en Japón son 3DS, PSV, PSP, PS3 y PS4. Algo que se destaca es que las plataformas portatiles son muy atractivas en esta región.

# Ahora realizemos un análisis descriptivo por géneros.

# In[77]:


# Agrupando los datos por genero y ventas NA
na_genre_sales = games_filtered.groupby('genre')['na_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
na_genre_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
na_genre_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
na_genre_sales.reset_index(inplace=True)
na_genre_sales


# In[78]:


# Agrupando los datos por genero y ventas EU
eu_genre_sales = games_filtered.groupby('genre')['eu_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
eu_genre_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
eu_genre_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
eu_genre_sales.reset_index(inplace=True)
eu_genre_sales


# In[79]:


# Agrupando los datos por genero y ventas JA
jp_genre_sales = games_filtered.groupby('genre')['jp_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
jp_genre_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
jp_genre_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
jp_genre_sales.reset_index(inplace=True)
jp_genre_sales


# In[80]:


# Comparando el desempeño de cada género sesgado por región. 
na_genre_sales['region'] = 'NA'
eu_genre_sales['region'] = 'EU'
jp_genre_sales['region'] = 'JP'

# Visualizando los resultados.
all_region_genre_sales = pd.concat([na_genre_sales[['genre', 'sales', 'region']], eu_genre_sales[['genre', 'sales', 'region']], jp_genre_sales[['genre', 'sales', 'region']]])

sns.barplot(data=all_region_genre_sales, x='genre', y='sales', hue='region', dodge=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Desempeño de cada región por género')
plt.xlabel('Género')
plt.ylabel('Ventas en millones de USD')
plt.legend(title='Región')
plt.xticks(rotation=90)
plt.show()


# La mayor cantidad de ventas en Norte América son generadas por los géneros Action, Sports y Shooter. En la región de Europa observamos las mismas tendencias en las ganancias, mientras que Japón se diferencia bastante ya que Role-Playing destaca sobre el resto y por poco iguala a Norte America, seguido por Action y Sports.
# 
# Action y Sports es un género destacado en ventas para las 3 regiones, algo resaltable.

# In[81]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=na_genre_sales, x='genre', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=na_genre_sales, x='genre', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(na_genre_sales['median_sales'].max(), na_genre_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Géneros potencialmente rentables en Norte America')
ax1.set_xlabel('Género')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Los 5 géneros potencialmente más rentables en Norte América son Platform, Shooter, Sports, Racing y Action.

# In[82]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=eu_genre_sales, x='genre', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=eu_genre_sales, x='genre', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(eu_genre_sales['median_sales'].max(), eu_genre_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Géneros potencialmente rentables en Europa')
ax1.set_xlabel('Género')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Los 5 géneros potencialmente más rentables en Europa son Shooter, Platform, Racing, Sports,  y Action. Son los mismos géneros que evidenciamos en Norte América, ambas regiones tienen comportamientos similares con algunas equeñas variaciones.

# In[83]:


# Visualizando los resultados.
# Crear la figura y el eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la mediana de ventas en el eje principal
sns.barplot(data=jp_genre_sales, x='genre', y='median_sales', color='blue', alpha=0.5, ax=ax1)
ax1.set_ylabel('Mediana de ventas (millones USD)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Crear un segundo eje Y
ax2 = ax1.twinx()

# Graficar la media de ventas en el segundo eje Y
sns.barplot(data=jp_genre_sales, x='genre', y='mean_sales', color='red', alpha=0.5, ax=ax2)
ax2.set_ylabel('Media de ventas (millones USD)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Establecer la misma escala en ambos ejes Y
max_value = max(jp_genre_sales['median_sales'].max(), jp_genre_sales['mean_sales'].max())
ax1.set_ylim(0, max_value * 1.1)  # Agregar un pequeño margen
ax2.set_ylim(0, max_value * 1.1)

# Configurar el título y ejes
plt.title('Géneros potencialmente rentables en Japón')
ax1.set_xlabel('Género')
ax1.tick_params("x", rotation=45)

# Mostrar la cuadrícula en el eje principal
ax1.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# Los 5 géneros potencialmente más rentables en Japón son Role-Playing, Fighting, Adventure, Platform, y Simulation, se diferencian parcialmente de las otras regiones.

# Finalmente para nuestro estudio revisaremos un análisis por clasificación y si estas tienen alguna influencia sobre las ventas regionales.

# In[84]:


# Agrupando los datos por rating y ventas NA
na_rating_sales = games_filtered.groupby('rating')['na_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
na_rating_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
na_rating_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
na_rating_sales.reset_index(inplace=True)
na_rating_sales


# In[85]:


# Agrupando los datos por rating y ventas EU
eu_rating_sales = games_filtered.groupby('rating')['eu_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
eu_rating_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
eu_rating_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
eu_rating_sales.reset_index(inplace=True)
eu_rating_sales


# In[86]:


# Agrupando los datos por rating y ventas JA
jp_rating_sales = games_filtered.groupby('rating')['jp_sales'].agg(['count', 'sum', 'mean', 'median','std'])

# Organizando los datos
jp_rating_sales.columns = ['games_released', 'sales', 'mean_sales', 'median_sales', 'std_sales']
jp_rating_sales.sort_values(['median_sales', 'mean_sales'], ascending=False, inplace=True)
jp_rating_sales.reset_index(inplace=True)
jp_rating_sales


# In[89]:


# Comparando el desempeño de cada clasificación sesgado por región.
na_rating_sales['region'] = 'NA'
eu_rating_sales['region'] = 'EU'
jp_rating_sales['region'] = 'JP'

# Visualizando los resultados
all_region_rating_sales = pd.concat([na_rating_sales[['rating', 'sales', 'region']], eu_rating_sales[['rating', 'sales', 'region']], jp_rating_sales[['rating', 'sales', 'region']]])

sns.barplot(data=all_region_rating_sales, x='rating', y='sales', hue='region', dodge=True)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Desempeño de cada región por clasificación')
plt.xlabel('Clasificación (ESRB)')
plt.ylabel('Ventas en millones de USD')
plt.legend(title='Clasificación')

plt.show()


# Notamos que las clasificaciones AO, K-A, EC y RP no tienen suficientes datos en ninguna región como para determinar alguna conclusion sobre su influencia en las ventas, es por eso que limitaremos nuestras conclusiones a las clasificaciones M, E, E10+ y T.
# 
# Las 3 regiones muestran mejores ventas en la clsificación E, tambien esta categoria es la que mayor cantidad de juegos registrados tiene puede estar relacionado con ello.
# 
# Si observamos las medianas de las clasificaciones relevantes nos damos cuenta que la clasificación M tiene mejor mediana en ventas en Norte America y Europa por lo que la determinaremos como la clasificación potencialmente mas rentable en estas. La diferencia la marca Japón quien muestra mejores tendencias de venta en su media para la clasificación T.
# 
# Estas conclusiones las confirma nuestra visualización la cual muestra comportamientos similares para Norte América y Europa, mientras que Japon describe mayores ventas en la clasificación T y menores en la M la cual es la segunda mas relevante en las otras regiones.

# ## Pruebas de hipótesis

# Realizaremos dos pruebas de hipótesis para nuestro estudio:
# 
# 1. Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# 
# 2. Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

# Para formular nuestra hipótesis nula describiremos la situación en donde no hay cambios en la comparación de la hipótesis, nuestra hipótesis alternativa por otro lado describira la situación en donde se persiben cambios en nuestra comparación.
# 
# Estableceremos el valor de umbral alfa como 0.05 para ambos casos.

# ### Prueba de hipótesis 1.

# **Hipótesis nula:** las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.
# 
# **Hipótesis alternativa:** las calificaciones promedio de los usuarios para las plataformas Xbox One y PC difieren.

# In[94]:


# Prueba la hipótesis
# Encontramos las calificaciones promedio de los usuarios para las plataformas XOne y PC, ignorando los valores nulos.
user_score_XOne = games_filtered[games_filtered['platform'] == 'XOne']['user_score'].dropna()
user_score_PC = games_filtered[games_filtered['platform'] == 'PC']['user_score'].dropna()

alpha = 0.05

resultados = st.ttest_ind(user_score_XOne, user_score_PC, equal_var=False)

print('valor p: ', resultados.pvalue)

# Comparamos el valor p con el umbral
if resultados.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# El valor p es increiblemente bajo, por lo que la posibilidad de que las calificaciones promedio provenientes de los usuarios para las plataformas XOne y PC sean iguales es muy baja, lo que nos permite rechazar la hipótesis nula.
# 
# Ya que rechazamos la hipótesis nula podemos responder a nuestra hipótesis original negandola y concluyendo que: las calificaciones promedio de los usuarios para las plataformas Xbox One y PC **NO** son las mismas.

# ### Pruebas de hipótesis 2.

# **Hipótesis nula:** las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son las mismas.
# 
# **Hipótesis alternativa:** las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.

# In[97]:


# Prueba la hipótesis
# Encontramos las calificaciones promedio de los usuarios para los generos Action y Sports, ignorando los valores nulos.
user_score_Action = games_filtered[games_filtered['genre'] == 'Action']['user_score'].dropna()
user_score_Sports = games_filtered[games_filtered['genre'] == 'Sports']['user_score'].dropna()

alpha = 0.05

resultados = st.ttest_ind(user_score_Action, user_score_Sports, equal_var=False)

print('valor p: ', resultados.pvalue)

# Comparamos el valor p con el umbral
if resultados.pvalue < alpha: 
    print("Rechazamos la hipótesis nula")
else:
    print("No podemos rechazar la hipótesis nula")


# El valor p (7.75%) es mayor al umbral que definimos (5%), lo que nos deja saber que no se puede descartar la hipótesis de que el promedio de las calificaciones para los dos generos podría ser igual.
# 
# Con esto en mente no podemos tampoco confirmar nuestra hipótesis original de que las calificaciones promedio de los usuarios para las plataformas Xbox One y PC difieren ya que hay un 7.75% de probabilidad de que estas no difieran y que debemos considerar.

# ## Conclusiones generales

# Después de realizar los analisis para los datos presentados podemos concluir los siguiente para nuestra campaña publicitaria del 2017:
# 
# - La campaña publicitaria para el 2017 debería enfocarse unicamente sobre las plataformas más recientes ya que son estas las que registran una mayor cantidad de ingresos historicamente. Estas plataformas son 'PS2', 'DS', 'PSP', 'PS3', 'PS4', 'PSV', '3DS', 'Wii', 'WiiU', 'X360', 'PC' y 'XOne'.
# 
# - Las campañas publicitarias deberían orientarse de forma diferente para las regiones estudiadas, en donde las regiones de Norte América y Europa pueden compartir campañas al tener comportamientos similares, sin embargo Japón debe tener otros enfoques diferentes basados en las tendencias observadas.
# 
# - La región de Norte América ha demostrado historicamente un mayor volumen de ventas por lo que se recomienda que la campaña  destine un mayor porcentaje de su presupuesto a esta región.
# 
# - Los géneros Action y Sports fueron géneros destacados en ventas para las 3 regiones, algo resaltable por lo que un enfoque en la campaña para estos dos generos podrían ser utilizado para todas las regiones y tener buenos resultados.
# 
# - Los juegos con una clasificación E tienen mejores registros de ventas en todas las regiones por lo que un enfoque en la campaña publicitaria para estos juegos podrían tener buenos resultados, como segunda opción los juegos con clasificación M se destacan tambien en Norte America y Europa, mientras que en Japón la segunda opción debería ser los juegos de clasificación T.
# 
# - En un enfoque regional personalizado para la campaña, Norte America debería enfocarse en juegos del género Shooter en la plataforma X360, Europa debería enfocarse en juegos del género Shooter en la plataforma PS4 y Japón debería enfocarse en el género Role-Playing en la plataforma 3DS. Adicionalmente se observo que Japón tienen mejores resultados en plataformas portatiles.
