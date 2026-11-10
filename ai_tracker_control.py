"""
Advanced AI Module for Solar Tracker Control
Intégration complète: Collecte météo → Prédiction PV → Optimisation angle
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. WEATHER DATA COLLECTOR & PROCESSOR
# ============================================================================

class WeatherDataCollector:
    """Collecte et traite les données météorologiques en temps réel"""
    
    def __init__(self):
        self.weather_data = {}
        self.data_history = pd.DataFrame()
        
    def collect_weather_data(self, irradiance, temperature, wind_speed, humidity, day_of_year):
        """Collecte les données météo actuelles"""
        self.weather_data = {
            'irradiance_kWh_m2': irradiance,      # Ensoleillement (kWh/m²)
            'temperature_C': temperature,         # Température (°C)
            'wind_speed_ms': wind_speed,          # Vitesse du vent (m/s)
            'humidity_percent': humidity,         # Humidité (%)
            'day_of_year': day_of_year           # Jour de l'année
        }
        return self.weather_data
    
    def compute_solar_angles(self, day_of_year, hour_of_day=12):
        """Calcule les angles solaires théoriques"""
        # Déclinaison solaire
        declination = 23.44 * np.sin(np.radians(360 * (day_of_year - 81) / 365))
        
        # Angle horaire (15° par heure)
        hour_angle = 15 * (hour_of_day - 12)
        
        # Latitude approximative (Sfax, Tunisie: ~34.7°N)
        latitude = 34.7
        
        # Hauteur solaire (elevation angle)
        elevation = np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        )
        elevation = np.max([np.degrees(elevation), 0])  # Min 0°
        
        # Azimut solaire
        azimuth = np.degrees(np.arctan2(
            np.sin(np.radians(hour_angle)),
            np.sin(np.radians(latitude)) * np.cos(np.radians(hour_angle)) -
            np.cos(np.radians(latitude)) * np.tan(np.radians(declination))
        ))
        
        return elevation, azimuth
    
    def get_weather_quality_score(self):
        """Score de qualité des conditions météo (0-100)"""
        irradiance = self.weather_data.get('irradiance_kWh_m2', 0)
        wind = self.weather_data.get('wind_speed_ms', 0)
        
        # Ensoleillement optimal: 6-8 kWh/m²
        irradiance_score = min((irradiance / 8) * 100, 100)
        
        # Vent acceptable: < 15 m/s
        wind_score = max(100 - (wind / 15) * 50, 50)
        
        # Score global
        quality_score = (irradiance_score * 0.7 + wind_score * 0.3)
        return min(quality_score, 100)


# ============================================================================
# 2. PV PRODUCTION PREDICTION ENGINE
# ============================================================================

class PVProductionPredictor:
    """Prédit la production PV basée sur les conditions météo"""
    
    def __init__(self, model=None, scaler=None, feature_cols=None):
        self.model = model or LinearRegression()
        self.scaler = scaler or StandardScaler()
        self.feature_cols = feature_cols or [
            'IRRADIANCE_kWh_m2', 'TEMPERATURE_C', 'WIND_SPEED_ms',
            'HUMIDITY_%', 'DAY_OF_YEAR'
        ]
        self.prediction_history = []
    
    def predict_pv_production(self, weather_data):
        """Prédit la production PV (kWh) selon les conditions météo"""
        
        # Prépare les features
        features = np.array([
            weather_data.get('irradiance_kWh_m2', 5),
            weather_data.get('temperature_C', 25),
            weather_data.get('wind_speed_ms', 4),
            weather_data.get('humidity_percent', 60),
            weather_data.get('day_of_year', 180)
        ]).reshape(1, -1)
        
        # Normalisation
        features_scaled = self.scaler.transform(features)
        
        # Prédiction
        pv_production = self.model.predict(features_scaled)[0]
        
        # Base de calcul: 1 kW/m² irradiance × surface × rendement
        # Hypothèse: 100m² de panneau, rendement 20%
        panel_area = 100  # m²
        efficiency = 0.20
        irradiance = weather_data.get('irradiance_kWh_m2', 0)
        
        # Production réelle estimée (kW)
        actual_production = (irradiance * panel_area * efficiency) * (1 + pv_production/100)
        
        self.prediction_history.append({
            'predicted_angle': pv_production,
            'estimated_production_kW': actual_production,
            'irradiance': irradiance
        })
        
        return {
            'predicted_angle': pv_production,
            'estimated_production_kW': max(actual_production, 0),
            'production_factor': pv_production / 100.0 if pv_production > 0 else 0
        }
    
    def get_production_forecast(self, forecast_hours=6):
        """Génère une prévision de production sur N heures"""
        if not self.prediction_history:
            return []
        
        forecast = []
        avg_production = np.mean([p['estimated_production_kW'] 
                                 for p in self.prediction_history[-10:]])
        
        for hour in range(forecast_hours):
            # Simulation simple: variation sinusoïdale
            variation = np.sin(np.pi * hour / (forecast_hours - 1))
            forecasted_kW = avg_production * max(variation, 0.3)
            forecast.append({
                'hour': hour,
                'forecasted_kW': forecasted_kW,
                'confidence': 0.95 - (hour * 0.05)
            })
        
        return forecast


# ============================================================================
# 3. OPTIMAL TRACKER ANGLE OPTIMIZER
# ============================================================================

class TrackerAngleOptimizer:
    """Optimise l'angle du suiveur pour maximiser la capture d'énergie"""
    
    def __init__(self):
        self.angle_history = []
        self.energy_capture = 0
        self.optimization_count = 0
    
    def calculate_optimal_angle(self, weather_data, elevation_angle, 
                               wind_penalty=True, temperature_factor=True):
        """
        Calcule l'angle optimal pour maximiser la capture d'énergie
        
        Paramètres:
        - weather_data: dict avec données météo
        - elevation_angle: hauteur du soleil (angle d'élévation)
        - wind_penalty: ajuste l'angle pour résister au vent
        - temperature_factor: ajuste pour la performance thermique
        """
        
        # Base: suivre la hauteur du soleil pour maximiser l'irradiance perpendiculaire
        optimal_angle = elevation_angle
        
        # Ajustement pour le vent (sécurité et stabilité)
        if wind_penalty:
            wind_speed = weather_data.get('wind_speed_ms', 0)
            # Réduire l'angle si vent fort (>12 m/s)
            if wind_speed > 12:
                wind_reduction = min((wind_speed - 12) * 2, 15)  # Max 15° de réduction
                optimal_angle = max(optimal_angle - wind_reduction, 0)
        
        # Ajustement pour la température (optimiser l'efficacité thermique)
        if temperature_factor:
            temp = weather_data.get('temperature_C', 25)
            # Efficiency decreases with temperature (approximately -0.5% per °C above 25°C)
            if temp > 25:
                temp_boost = (temp - 25) * 0.1  # Léger ajustement
                optimal_angle = optimal_angle + temp_boost
        
        # Limites physiques (0° à 90°)
        optimal_angle = np.clip(optimal_angle, 0, 90)
        
        # Calcul de l'énergie capturée
        energy_factor = np.sin(np.radians(optimal_angle))
        irradiance = weather_data.get('irradiance_kWh_m2', 0)
        energy_captured = energy_factor * irradiance
        
        self.angle_history.append({
            'angle': optimal_angle,
            'elevation': elevation_angle,
            'wind_speed': weather_data.get('wind_speed_ms', 0),
            'temperature': weather_data.get('temperature_C', 0),
            'energy_captured': energy_captured
        })
        
        self.energy_capture += energy_captured
        self.optimization_count += 1
        
        return {
            'optimal_angle': optimal_angle,
            'elevation_angle': elevation_angle,
            'energy_factor': energy_factor,
            'energy_captured': energy_captured,
            'confidence': self._calculate_confidence(weather_data)
        }
    
    def _calculate_confidence(self, weather_data):
        """Calcule le score de confiance de l'optimisation (0-1)"""
        irradiance = weather_data.get('irradiance_kWh_m2', 0)
        wind_speed = weather_data.get('wind_speed_ms', 0)
        
        # Confiance élevée avec bon ensoleillement et vent modéré
        irradiance_confidence = min(irradiance / 8, 1.0)  # Optimal à 8 kWh/m²
        wind_confidence = max(1 - (wind_speed / 20), 0)   # Réduit si vent fort
        
        confidence = (irradiance_confidence * 0.7 + wind_confidence * 0.3)
        return max(min(confidence, 1.0), 0.3)  # Entre 0.3 et 1.0
    
    def get_efficiency_metrics(self):
        """Retourne les métriques d'efficacité"""
        if self.optimization_count == 0:
            return {}
        
        angles = [h['angle'] for h in self.angle_history]
        energy_captured = [h['energy_captured'] for h in self.angle_history]
        
        return {
            'avg_angle': np.mean(angles),
            'angle_std': np.std(angles),
            'total_energy_captured': self.energy_capture,
            'avg_energy_per_optimization': self.energy_capture / self.optimization_count,
            'optimization_count': self.optimization_count
        }
    
    def compare_strategies(self, weather_data):
        """Compare différentes stratégies de suiveur"""
        irradiance = weather_data.get('irradiance_kWh_m2', 0)
        elevation = 45  # Angle d'élévation constant
        
        # Stratégie 1: Angle fixe (30°)
        fixed_angle = 30
        energy_fixed = np.sin(np.radians(fixed_angle)) * irradiance
        
        # Stratégie 2: Suivi classique saisonnier
        day_of_year = weather_data.get('day_of_year', 180)
        seasonal_angle = 45 + 35 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        seasonal_angle = np.clip(seasonal_angle, 0, 90)
        energy_seasonal = np.sin(np.radians(seasonal_angle)) * irradiance
        
        # Stratégie 3: Suivi intelligent (IA optimisée)
        optimal_result = self.calculate_optimal_angle(weather_data, elevation)
        ai_angle = optimal_result['optimal_angle']
        energy_ai = optimal_result['energy_captured']
        
        # Calcul des gains
        gain_ai_vs_fixed = ((energy_ai - energy_fixed) / max(energy_fixed, 0.1)) * 100 if energy_fixed > 0 else 0
        gain_ai_vs_seasonal = ((energy_ai - energy_seasonal) / max(energy_seasonal, 0.1)) * 100 if energy_seasonal > 0 else 0
        
        return {
            'fixed_angle': fixed_angle,
            'energy_fixed': energy_fixed,
            'seasonal_angle': seasonal_angle,
            'energy_seasonal': energy_seasonal,
            'optimal_angle': ai_angle,
            'energy_optimal': energy_ai,
            'gain_vs_fixed_percent': gain_ai_vs_fixed,
            'gain_vs_seasonal_percent': gain_ai_vs_seasonal
        }


# ============================================================================
# 4. INTEGRATED AI CONTROL SYSTEM
# ============================================================================

class SolarTrackerAISystem:
    """Système intégré de contrôle du suiveur solaire avec IA"""
    
    def __init__(self, model=None, scaler=None, feature_cols=None):
        self.weather_collector = WeatherDataCollector()
        self.pv_predictor = PVProductionPredictor(model, scaler, feature_cols)
        self.angle_optimizer = TrackerAngleOptimizer()
        self.control_log = []
    
    def process_real_time_update(self, irradiance, temperature, wind_speed, 
                                 humidity, day_of_year, hour_of_day=12):
        """
        Traitement complet et en temps réel:
        Météo → Prédiction PV → Optimisation angle
        """
        
        # 1. Collecte météo
        weather_data = self.weather_collector.collect_weather_data(
            irradiance, temperature, wind_speed, humidity, day_of_year
        )
        
        # 2. Calcul des angles solaires
        elevation, azimuth = self.weather_collector.compute_solar_angles(
            day_of_year, hour_of_day
        )
        
        # 3. Prédiction de production PV
        pv_prediction = self.pv_predictor.predict_pv_production(weather_data)
        
        # 4. Optimisation de l'angle
        optimization = self.angle_optimizer.calculate_optimal_angle(weather_data, elevation)
        
        # 5. Comparaison des stratégies
        comparison = self.angle_optimizer.compare_strategies(weather_data)
        
        # 6. Score de qualité
        quality_score = self.weather_collector.get_weather_quality_score()
        
        # Enregistrement
        control_event = {
            'weather_data': weather_data,
            'solar_angles': {'elevation': elevation, 'azimuth': azimuth},
            'pv_prediction': pv_prediction,
            'optimization': optimization,
            'strategy_comparison': comparison,
            'quality_score': quality_score
        }
        
        self.control_log.append(control_event)
        
        return control_event
    
    def get_system_status(self):
        """Retourne le statut complet du système"""
        if not self.control_log:
            return None
        
        latest = self.control_log[-1]
        metrics = self.angle_optimizer.get_efficiency_metrics()
        
        return {
            'weather_data': latest['weather_data'],
            'solar_angles': latest['solar_angles'],
            'optimization': latest['optimization'],
            'pv_prediction': latest['pv_prediction'],
            'strategy_comparison': latest['strategy_comparison'],
            'quality_score': latest['quality_score'],
            'current_energy_gain': latest['strategy_comparison']['gain_vs_fixed_percent'],
            'pv_production_forecast': self.pv_predictor.get_production_forecast(),
            'efficiency_metrics': metrics,
            'timestamp': pd.Timestamp.now()
        }
