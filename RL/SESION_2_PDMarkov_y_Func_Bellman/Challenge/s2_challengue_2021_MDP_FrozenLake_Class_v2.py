import gym
import numpy as np




def value_iterator(env, max_iterations=100000, gamma=0.9):
    stateValue = [0 for i in range( env.nS )] # env.nS -> Número de estados. Lista con los posibles estados que se inicializan con valor cero
    newStateValue = stateValue.copy() # Lista que guarda el nuevo valor de cada estado
    for i in range( max_iterations ):
#         print('Iteration: ',i)
        for state in range( env.nS ): # Iteramos por cada estado.
            action_values = []
            for action in range( env.nA): #env.nA -> Número de acciones posibles
                state_value = 0
                for i in range( len( env.P[state][action])): # env.P es una lista que contiene todos los estados, donde cada estado a su vez contiene un diccionario que mapea todas las acciones posibles del estado al siguiente estado si tomamos esa accion, Probabilidad de ir al estado siguiente, reward, y si el juego termina o no.
                    # Te recomiendo que te imprimas en una celda a parte env.P, para su mejor comprensión#
                    prob, next_state, reward, done = env.P[state][action][i]
                    state_action_value = prob*(reward+gamma*stateValue[next_state]) # Cálculo de del valor estado-acción (fórmula).
                    state_value += state_action_value
                action_values.append( state_value )  # el valor de cada acción
                best_action = np.argmax(np.asarray(action_values))  # elegimos la acción que da el máximo valor
                newStateValue[state] = action_values[best_action] ##YOUR CODE HERE##  # actulizamos el valor para cada estado con el de la mejor acción
#                 print(f'Nuevo estado {newStateValue[state]}')
#                 print(f'Nuevo estado {newStateValue}')
                

        if i > 1000:
            if sum( stateValue ) - sum( newStateValue ) < 1e-04:  # Si la diferencia es menor que el umbral fijado, detenemos el bucle.
                break
                print('Last iteration:', str(i))
        else:
            stateValue = newStateValue.copy()
    return stateValue



env = gym.make( 'FrozenLake-v0', is_slippery=False)
print(env.P[0])
print(env.P)
print(env.nS)
print(env.nA)


# ### 2. Obteniendo la política a partir del valor del estado.

# Ahora que ya hemos calculado la función de valor para todos los estados, el siguiente paso será extraer la política a partir de la función de valor. Lo haremos de una forma similar, para cada estado concreto, se calcula el par de valor estado-acción de todas las acciones posibles en ese estado y se elige la acción con el mayor valor estado-acción.
# 
# Completa la función *get_policiy*
# 

# In[10]:


def get_policy(env,stateValue, gamma=0.9):
  policy = [0 for i in range(env.nS)] # Lista de longitud número de estados inicializada a cero.
  for state in range(env.nS):## Número de estados ##): 
    action_values = []
    for action in range(env.nA):## Número de acciones ##):
      action_value = 0
      for i in range(len(env.P[state][action])):
        prob, next_state, r, _ =  env.P[state][action][i]## acción posible para cada estado ##
        action_value += prob*(r+gamma*stateValue[next_state]) ## Valor de la acción ##
      action_values.append(action_value)
    best_action = np.argmax(np.asarray(action_values))##YOUR CODE HERE##  # elegimos la acción que da el máximo valor
    policy[state] = best_action##YOUR CODE HERE## # Actualizamos la política con la mejor acción.
  return policy 




def get_score_and_best_path(env, policy, episodes=1000):
    misses = 0
    steps_list = []
    actions_walked = []
    for episode in range(episodes):## número de episodios ##:
        observation =  env.reset()## Estado inicial##
        action_list = []
        steps = 0
        while True:

            action = policy[observation]##acción de la política para cada estado u observación## 
            observation, reward, done, _ = env.step(action)## acción ##)
            action_list.append(action)
            steps += 1
            if done and reward == 1:## YOUR CODE HERE##:  # Si el episodio ha llegado al final y ha conseguido el objetivo
                print('You have got the Frisbee after {} steps'.format(steps))
                steps_list.append( steps )
                actions_walked.append(action_list)
                break
            elif done and reward == 0:## YOUR CODE HERE##: #Si el episodio ha llegado al final y ha caído por un agujero
                print("You fell in a hole!")
                misses += 1
                break

    print( '----------------------------------------------' )
    print( 'You took an average of {:.0f} steps to get the frisbee'.format( np.mean( steps_list ) ) )
    print( 'And you fell in the hole {:.2f} % of the times'.format( (misses / episodes) * 100 ) )
    print( '----------------------------------------------' )

    best_way = actions_walked[0] 
    for i in actions_walked:
        if len(i)<len(best_way): #Elegimos la iteracción con el menor número de pasos
            best_way = i
        else:
            pass

    return best_way

def show_path(best_way):
    
    print('Number of steps:', str(len(best_way)))
    print("Best way:", str(best_way))
    
    env.reset()
    env.render()
    
    for i in best_way:
        new_state, reward, done, info = env.step(i)
        env.render()
        if done:
            break
    env.close()

env = gym.make('FrozenLake8x8-v0', is_slippery=False) # is_slippery=True (default) -> Entorno estocástico
## YOUR CODE HERE ## # Inicializamos el entorno

observation = env.reset()
previous_observation = observation

stateValues = value_iterator(env,max_iterations=100)## YOUR CODE HERE ## # Calculamos los valores de estado hasta 100000 iteraciones
policy = get_policy(env, stateValues)## YOUR CODE HERE ## # Obtenemos la política para los valores de estado
print(policy)
print(len(policy))
best_path= get_score_and_best_path(env, policy,episodes=10)## YOUR CODE HERE ## Calculamos el mejor recorrido

# ## YOUR CODE HERE ## # Visualizamos el mejor recorrido llevado a cabo
show_path(best_path)


# #### Ahora vuelve a entrenar al agente en el entorno estocástico y comenta las diferencias que observes.

# In[14]:


env_esto = gym.make('FrozenLake8x8-v0', is_slippery=True) # is_slippery=True (default) -> Entorno estocástico
## YOUR CODE HERE ## # Inicializamos el entorno

observation = env_esto.reset()
previous_observation = observation

stateValues = value_iterator(env_esto,max_iterations=100)## YOUR CODE HERE ## # Calculamos los valores de estado hasta 100000 iteraciones
policy = get_policy(env_esto, stateValues)## YOUR CODE HERE ## # Obtenemos la política para los valores de estado
print(policy)
print(len(policy))
best_path= get_score_and_best_path(env_esto, policy,episodes=10)## YOUR CODE HERE ## Calculamos el mejor recorrido

## YOUR CODE HERE ## # Visualizamos el mejor recorrido llevado a cabo
show_path(best_path)


# In[ ]:




