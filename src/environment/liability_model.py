import numpy as np
import pandas as pd

class LiabilityModel:
    """
    Modela y valora los pasivos de un plan de pensión.
    
    El valor de los pasivos cambia dinámicamente en función
    de la tasa de descuento (vinculada a las tasas de CETES).
    """

    def __init__(self, initial_cash_flows: pd.DataFrame):
        """
        Inicializa el modelo con un conjunto fijo de flujos de caja futuros.
        
        Args:
            initial_cash_flows (pd.DataFrame): Un DataFrame con una columna 'cash_flow'
                                               que representa los pagos anuales.
        """
        self.initial_cash_flows = initial_cash_flows.copy()
        self.current_cash_flows = self.initial_cash_flows.copy()
        self.present_value = 0.0
        self.current_step = 0
        self.num_periods = len(self.initial_cash_flows)
    
    def step(self, discount_rate: float, wage_inflation_rate: float) -> float:
        """
        Avanza un paso en el tiempo, actualizando el valor de los pasivos.
        
        Args:
            discount_rate (float): La tasa de interés actual del mercado, usada
                                   como la tasa de descuento.
            wage_inflation_rate (float): Tasa de inflación salarial para ajustar
                                         los flujos de caja futuros.
                                         
        Returns:
            float: El valor presente actualizado de los pasivos.
        """
        if self.current_step >= self.num_periods:
            self.present_value = 0.0
            return self.present_value
            
        # 1. Actualizar los flujos de caja restantes con la inflación salarial
        # Esto refleja que los beneficios futuros crecen con los salarios.
        self.current_cash_flows.iloc[self.current_step:] *= (1 + wage_inflation_rate)
        
        # 2. Revalorizar los pasivos restantes en base a la nueva tasa de descuento
        remaining_cash_flows = self.current_cash_flows.iloc[self.current_step:]
        periods = np.arange(1, len(remaining_cash_flows) + 1)
        
        # Fórmula de Valor Presente
        pv = np.sum(remaining_cash_flows.values.flatten() / ((1 + discount_rate) ** periods))
        
        self.present_value = pv
        self.current_step += 1
        return self.present_value
        
    def reset(self, initial_discount_rate: float) -> float:
        """
        Reinicia el modelo de pasivos para una nueva simulación.
        """
        self.current_cash_flows = self.initial_cash_flows.copy()
        self.current_step = 0
        self.present_value = self._calculate_initial_pv(initial_discount_rate)
        return self.present_value

    def _calculate_initial_pv(self, discount_rate: float) -> float:
        """Calcula el valor presente inicial."""
        periods = np.arange(1, self.num_periods + 1)
        pv = np.sum(self.initial_cash_flows.values.flatten() / ((1 + discount_rate) ** periods))
        return pv