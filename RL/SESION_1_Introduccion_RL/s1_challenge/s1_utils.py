def env_wrapper(policy_func, env):
    '''
    Simula el entorno `env` siguiendo la política `policy_func`.
    '''
    # Reseteamos el entorno
    observation = env.reset()
    # al empezar, tomamos la observación anterior como la primera
    previous_observation = observation

    # Ejecutamos 250 pasos en el entorno
    try:
        for step in range(250):

            # Visualizamos el juego de MountainCar
            env.render()

            # Tomamos una acción siguiendo policy_func
            action = policy_func(observation, step, previous_observation)

            # actualizamos previous_observation
            previous_observation = observation
            
            # Ejecutamos la acción
            observation, reward, done, info = env.step(action)
            
            if done:
                print(f'Enhorabuena! Lo has logrado en {step} pasos')
                break
        else:
            print(f'Lo siento! No has alcanzado el objetivo')
    except Exception as e:
        raise e
    finally:
        env.close()