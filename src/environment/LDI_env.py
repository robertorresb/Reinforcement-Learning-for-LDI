import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from .liability_model import LiabilityModel
import math

class LDI_env(gym.Env):
    """
    Entorno de simulación de Inversión Impulsada por Pasivos (LDI).
    
    Basado en el paper y el script de Kailan Shang para modelar un ambiente
    dinámico donde los pasivos y activos se revalorizan en cada paso.
    """
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        initial_sp500_ratio: float = 0.5,
        episode_length_months: int = 120, # 10 años, en línea con el paper
        commission_rate: float = 0.001,
        render_mode: str | None = None,
    ):
        super().__init__()
        
        # Parámetros del ambiente
        self.initial_capital = initial_capital
        self.initial_sp500_ratio = initial_sp500_ratio
        self.episode_length_months = episode_length_months
        self.commission_rate = commission_rate
        self.render_mode = render_mode
        
        # ESG simplificado: Parámetros de simulación estocástica
        self.sp500_mu = 0.008 # Rendimiento promedio mensual
        self.sp500_sigma = 0.05 # Volatilidad mensual
        self.cetes_mu = 0.005 # Tasa de rendimiento promedio mensual
        self.cetes_sigma = 0.005
        self.inflation_mu = 0.003 # Inflación salarial promedio mensual
        self.inflation_sigma = 0.002
        
        # Matriz de correlación (simplificada)
        # S&P500 vs. CETES vs. Inflación Salarial
        self.corr_matrix = np.array([
            [1.0, -0.4, 0.3],  # S&P500 se correlaciona negativamente con CETES (aumento de tasas)
            [-0.4, 1.0, 0.5],  # CETES se correlaciona positivamente con la inflación
            [0.3, 0.5, 1.0]
        ])
        
        # Definición de los pasivos (ejemplo simple a 30 años)
        liability_payments = np.full(360, 100000.0) # 360 meses = 30 años
        self.initial_liabilities = pd.DataFrame(data=liability_payments, columns=['cash_flow'])
        
        # Instancia del modelo de pasivos
        self.liability_model = LiabilityModel(self.initial_liabilities)
        
        # Espacio de acciones: peso en S&P 500 (renta variable)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Espacio de observación: [funding_ratio, sp500_return, cetes_return, current_sp500_ratio]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Variables de estado
        self.current_step = 0
        self.current_portfolio_value = 0.0
        self.sp500_shares = 0.0
        self.cetes_value = 0.0
        self.liability_present_value = 0.0
        self.last_sp500_price = 100.0 # Precio inicial arbitrario
        
        # Historial para análisis
        self.historical_data = []

    def _generate_economic_scenario(self):
        """
        Genera rendimientos mensuales correlacionados para los factores económicos.
        Esto simula un ESG simplificado.
        """
        # Generar variables aleatorias con la matriz de correlación
        rand_vars = np.random.multivariate_normal(
            [0, 0, 0], self.corr_matrix
        )
        
        sp500_return = self.sp500_mu + self.sp500_sigma * rand_vars[0]
        cetes_rate = self.cetes_mu + self.cetes_sigma * rand_vars[1]
        inflation_rate = self.inflation_mu + self.inflation_sigma * rand_vars[2]
        
        return sp500_return, cetes_rate, inflation_rate

    def reset(self, seed: int | None = None):
        """
        Reinicia el ambiente para un nuevo episodio de simulación.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Reiniciar el modelo de pasivos
        initial_cetes_rate = self.cetes_mu
        self.liability_model.reset(initial_cetes_rate)
        self.liability_present_value = self.liability_model.present_value
        
        # Inicializar el portafolio
        initial_sp500_capital = self.initial_capital * self.initial_sp500_ratio
        initial_cetes_capital = self.initial_capital * (1 - self.initial_sp500_ratio)
        
        self.sp500_shares = initial_sp500_capital / self.last_sp500_price
        self.cetes_value = initial_cetes_capital
        self.current_portfolio_value = self.initial_capital
        
        # Reiniciar historial
        self.historical_data = []
        
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Avanza un paso de tiempo (un mes) en el ambiente.
        """
        # Asegurar que la acción sea válida (entre 0 y 1)
        target_sp500_ratio = np.clip(action[0], 0.0, 1.0)
        
        # 1. Generar nuevo escenario económico para el mes
        sp500_return, cetes_rate, inflation_rate = self._generate_economic_scenario()
        
        # 2. Actualizar el valor de los activos
        self.last_sp500_price *= (1 + sp500_return)
        self.current_portfolio_value = (self.sp500_shares * self.last_sp500_price) + (self.cetes_value * (1 + cetes_rate))
        
        # 3. Rebalancear el portafolio
        rebalance_cost = 0.0
        
        target_sp500_value = self.current_portfolio_value * target_sp500_ratio
        
        if (self.sp500_shares * self.last_sp500_price) > target_sp500_value:
            # Vender S&P500 y transferir a CETES
            sell_amount = (self.sp500_shares * self.last_sp500_price) - target_sp500_value
            self.cetes_value += sell_amount * (1 - self.commission_rate)
            rebalance_cost = sell_amount * self.commission_rate
        else:
            # Comprar S&P500 y transferir de CETES
            buy_amount = target_sp500_value - (self.sp500_shares * self.last_sp500_price)
            if self.cetes_value >= buy_amount:
                self.cetes_value -= buy_amount * (1 + self.commission_rate)
                rebalance_cost = buy_amount * self.commission_rate
            else:
                # Si no hay suficiente efectivo, compra lo que puede
                buy_amount = self.cetes_value / (1 + self.commission_rate)
                self.cetes_value = 0
                rebalance_cost = buy_amount * self.commission_rate
        
        self.sp500_shares = (self.current_portfolio_value * target_sp500_ratio) / self.last_sp500_price
        
        # 4. Actualizar el valor de los pasivos
        self.liability_present_value = self.liability_model.step(cetes_rate, inflation_rate)
        
        # 5. Calcular la recompensa
        reward, done = self._calculate_reward()
        
        # 6. Guardar historial
        self.historical_data.append(self._get_info())
        
        self.current_step += 1
        
        # La simulación termina si se alcanza la duración del episodio o si hay un déficit severo
        done = done or self.current_step >= self.episode_length_months
        
        return self._get_observation(), reward, done, False, self._get_info()

    def _calculate_reward(self):
        """
        Calcula la recompensa del agente.
        """
        # La recompensa principal es el ratio de financiamiento
        if self.liability_present_value <= 0:
            funding_ratio = float('inf')
        else:
            funding_ratio = self.current_portfolio_value / self.liability_present_value

        # Recompensa base
        reward = 0.0
        
        # Recompensa por mantener un ratio alto, con un piso de 1.0
        reward += (funding_ratio - 1.0) * 1000.0 if funding_ratio > 1.0 else (funding_ratio - 1.0) * 5000.0
        
        done = False
        # Penalización por déficit: El agente "muere" si el déficit es demasiado grande
        if funding_ratio < 0.8:
            reward = -10000.0
            done = True
        
        # Penalización por alto riesgo (opcional, ajusta según el objetivo)
        # Esto penaliza si el agente pone demasiado en renta variable sin una buena razón
        current_sp500_value = self.sp500_shares * self.last_sp500_price
        current_sp500_ratio = current_sp500_value / self.current_portfolio_value
        if current_sp500_ratio > 0.8 and funding_ratio < 1.1:
            reward -= 50.0

        return reward, done

    def _get_observation(self):
        """
        Devuelve el estado actual del ambiente.
        """
        # Normalizar para un mejor aprendizaje
        if self.liability_present_value > 0:
            funding_ratio = self.current_portfolio_value / self.liability_present_value
        else:
            funding_ratio = 1.0 # o un valor grande, si no hay pasivos
            
        sp500_value = self.sp500_shares * self.last_sp500_price
        current_sp500_ratio = sp500_value / self.current_portfolio_value
            
        observation = np.array([
            funding_ratio,
            sp500_value / self.initial_capital, # Valor de activos de riesgo normalizado
            self.cetes_value / self.initial_capital, # Valor de activos de bajo riesgo normalizado
            current_sp500_ratio, # Composición del portafolio
        ], dtype=np.float32)
        
        return observation

    def _get_info(self) -> dict:
        """
        Devuelve un diccionario con información adicional.
        """
        sp500_value = self.sp500_shares * self.last_sp500_price
        current_sp500_ratio = sp500_value / self.current_portfolio_value
        
        return {
            'funding_ratio': self.current_portfolio_value / self.liability_present_value,
            'portfolio_value': self.current_portfolio_value,
            'liabilities_value': self.liability_present_value,
            'sp500_price': self.last_sp500_price,
            'sp500_ratio': current_sp500_ratio,
            'cetes_value': self.cetes_value,
            'current_step': self.current_step
        }

    def render(self, mode="human") -> None:
        if mode == "human":
            info = self._get_info()
            print("--- Estado Actual ---")
            print(f"Paso: {info['current_step']} / {self.episode_length_months}")
            print(f"Valor del Portafolio: ${info['portfolio_value']:,.2f} MXN")
            print(f"Valor de los Pasivos: ${info['liabilities_value']:,.2f} MXN")
            print(f"**Ratio de Financiamiento: {info['funding_ratio']:.4f}**")
            print(f"Composición (S&P500): {info['sp500_ratio']:.2f}")
            print("-" * 20)

    def close(self) -> None:
        print("Ambiente LDI cerrado.")