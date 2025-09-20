import pandas as pd
import numpy as np

class LiabilityModel:
    """
    Modela los pasivos de un plan de pensión con flujos de caja dinámicos.
    """
    def __init__(self, cash_flow_df, initial_discount_rate):
        """
        Inicializa el modelo de pasivos.

        Args:
            cash_flow_df (pd.DataFrame): DataFrame con la columna 'cash_flow' y la fecha como índice.
            initial_discount_rate (float): Tasa de interés inicial para calcular el valor presente.
        """
        self.cash_flows = cash_flow_df
        self.initial_discount_rate = initial_discount_rate
        self.current_step = 0
        self.present_value = self._calculate_present_value(initial_discount_rate)

    def _calculate_present_value(self, discount_rate):
        """
        Calcula el valor presente de los flujos de caja restantes.
        
        Este cálculo se revaloriza en cada paso, como se menciona en tu documento de referencia.

        Args:
            discount_rate (float): La tasa de interés actual del mercado.

        Returns:
            float: El valor presente total de los pasivos.
        """
        remaining_cash_flows = self.cash_flows.iloc[self.current_step:]
        periods = np.arange(1, len(remaining_cash_flows) + 1)
        
        # Fórmula simplificada: Sumatoria de (Pagos Futuros / (1 + Tasa de Descuento)^t)
        present_value = np.sum(remaining_cash_flows.values.flatten() / (1 + discount_rate)**periods)
        return present_value
    
    def step(self, new_discount_rate, wage_inflation_rate):
        """
        Avanza un paso en la simulación, actualizando el valor de los pasivos.
        
        Args:
            new_discount_rate (float): La nueva tasa de interés para descontar.
            wage_inflation_rate (float): La tasa de inflación salarial para ajustar los pagos futuros.

        Returns:
            float: El valor presente actualizado de los pasivos.
        """
        # Actualiza los flujos de caja futuros con la inflación salarial
        # Esto refleja la "Revalorización de los Pasivos" en tu documento.
        self.cash_flows.iloc[self.current_step:] *= (1 + wage_inflation_rate)
        
        self.current_step += 1
        self.present_value = self._calculate_present_value(new_discount_rate)
        return self.present_value

    def reset(self):
        """
        Reinicia el modelo de pasivos para una nueva simulación.
        """
        self.current_step = 0
        self.present_value = self._calculate_present_value(self.initial_discount_rate)

# ---
# Ejemplo de uso:
if __name__ == '__main__':
    # 1. Definir los flujos de caja iniciales de los pasivos (como un plan a 30 años)
    num_years = 30
    initial_payout = 10_000_000 # MXN
    
    dates = pd.date_range(start='2025-01-01', periods=num_years, freq='A')
    cash_flows = pd.DataFrame(data=initial_payout, index=dates, columns=['cash_flow'])

    # 2. Simular tasas de interés y de inflación salarial
    # En tu entorno real, estas tasas provendrán de tus datos de CETES.
    simulated_interest_rates = np.random.uniform(low=0.03, high=0.07, size=num_years)
    simulated_wage_inflation = np.random.uniform(low=0.02, high=0.04, size=num_years)

    # 3. Inicializar el modelo
    initial_rate = simulated_interest_rates[0]
    liability_model = LiabilityModel(cash_flows, initial_rate)
    
    print(f"Valor inicial de los pasivos: ${liability_model.present_value:,.2f} MXN")

    # 4. Simular el paso del tiempo
    for i in range(1, num_years):
        new_rate = simulated_interest_rates[i]
        new_inflation = simulated_wage_inflation[i]
        updated_pv = liability_model.step(new_rate, new_inflation)
        print(f"Año {i+1} - Tasa de interés: {new_rate*100:.2f}% - Inflación salarial: {new_inflation*100:.2f}% - Valor actual de los pasivos: ${updated_pv:,.2f} MXN")