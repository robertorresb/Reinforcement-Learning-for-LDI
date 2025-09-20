import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

# Ajusta la ruta del proyecto para importar el ambiente
# Esto permite que el script se ejecute desde cualquier lugar dentro del proyecto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Importa tu entorno LDI
from src.environment.LDI_env import LDI_env

def train_ldi_model(
    log_dir="logs/tensorboard",
    checkpoints_dir="checkpoints",
    best_model_dir="models/best_model",
    eval_log_dir="logs/evaluation",
    total_timesteps=5_000_000,
    episode_length_months=120, 
    initial_capital=100_000_000,
    initial_sp500_ratio=0.5,
    commission=0.001,
    algorithm="PPO",
    verbose=True,
    n_envs=4
):
    """
    Entrena un agente de Reinforcement Learning para la gestión de LDI.
    
    Args:
        log_dir: Directorio para logs de TensorBoard.
        checkpoints_dir: Directorio para guardar checkpoints del modelo.
        best_model_dir: Directorio para guardar el mejor modelo.
        eval_log_dir: Directorio para logs de evaluación.
        total_timesteps: Número total de pasos de tiempo para entrenar.
        episode_length_months: Duración de cada episodio en meses.
        initial_capital: Capital inicial para el portafolio.
        initial_sp500_ratio: Composición inicial del portafolio.
        commission: Tasa de comisión por transacción.
        algorithm: Algoritmo de RL a usar ("PPO" o "DDPG").
        n_envs: Número de ambientes paralelos para el entrenamiento.
        verbose: Si se deben imprimir los mensajes de progreso.
    
    Returns:
        model: El modelo de RL entrenado.
    """
    # Crear directorios si no existen
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # --- NOTA IMPORTANTE ---
    # En esta versión, el ambiente LDI_env simula los retornos de activos y las tasas
    # de interés de forma estocástica (usando un ESG simplificado).
    # Por lo tanto, no es necesario cargar los datos de los archivos CSV aquí.
    # Los datos de los pasivos (flujos de caja) se crean directamente en el script
    # para simplificar. En un caso real, esto provendría de tu modelo actuarial.
    
    # Crear los flujos de caja de los pasivos para el ambiente
    liability_payments = pd.DataFrame({
        'cash_flow': np.full(360, 10000.0) # Ejemplo: pagos mensuales por 30 años
    })
    
    if verbose:
        print("Creando ambientes de entrenamiento...")
    
    def make_env():
        """Función helper para crear una instancia del entorno."""
        return LDI_env(
            initial_capital=initial_capital,
            initial_sp500_ratio=initial_sp500_ratio,
            episode_length_months=episode_length_months,
            commission_rate=commission,
        )
    
    # Crear el ambiente vectorizado para el entrenamiento en paralelo
    train_env = make_vec_env(lambda: make_env(), n_envs=n_envs, seed=0)
    
    # Crear un ambiente de evaluación (sin vectorizar)
    eval_env = LDI_env(
        initial_capital=initial_capital,
        initial_sp500_ratio=initial_sp500_ratio,
        episode_length_months=episode_length_months,
        commission_rate=commission,
    )
    
    # Configurar el logger para TensorBoard
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    
    if algorithm.upper() == "DDPG":
        if verbose:
            print("Usando DDPG (Deep Deterministic Policy Gradient) para LDI...")
        
        n_actions = train_env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions) 
        )
        
        model = DDPG(
            "MlpPolicy",
            train_env,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
        )
        model_prefix = "ddpg_ldi"
        
    else:  # PPO
        if verbose:
            print("Usando PPO (Proximal Policy Optimization) para LDI...")
        
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.999,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])
        )
        model_prefix = "ppo_ldi"
    
    model.set_logger(new_logger)
    
    # Configurar los callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path=checkpoints_dir,
        name_prefix=model_prefix
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Entrenar el modelo
    if verbose:
        print(f"Entrenando {algorithm} por {total_timesteps} pasos de tiempo...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=f"{model_prefix}_run"
    )
    
    # Guardar el modelo final
    final_model_path = os.path.join("models", f"{model_prefix}_final")
    model.save(final_model_path)
    if verbose:
        print(f"Modelo final guardado en {final_model_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de LDI')
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG'], default='PPO',
                        help='Algoritmo de RL a usar')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Número total de pasos de tiempo de entrenamiento')
    parser.add_argument('--episode_months', type=int, default=120,
                        help='Duración de cada episodio en meses')
    
    args = parser.parse_args()
    
    print("Iniciando el entrenamiento del modelo de LDI...")
    trained_model = train_ldi_model(
        total_timesteps=args.timesteps,
        episode_length_months=args.episode_months,
        algorithm=args.algorithm
    )