"""
F1 2026 Season Prediction Model - ENHANCED VERSION
HEAVILY weighted on 2024-2025 performance (70%) + regulation change patterns (30%)
"""

import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Enable FastF1 cache
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

class F1_2026_Predictor_Enhanced:
    """
    Enhanced ML model prioritizing 2024-2025 recent performance
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
        # Core data structures
        self.team_2025_performance = {}
        self.driver_2025_performance = {}
        self.team_trajectory = {}
        self.regulation_adaptation_scores = self._init_regulation_scores()
        self.power_unit_config = self._init_power_units()
        self.track_characteristics = self._init_track_data()
        
    def _init_regulation_scores(self):
        """
        Regulation change adaptation - 30% weight only
        """
        return {
            'Red Bull Racing': 8.5, 'Ferrari': 7.0, 'Mercedes': 9.0,
            'McLaren': 7.5, 'Aston Martin': 6.5, 'Alpine': 5.5,
            'Williams': 5.0, 'RB': 6.0, 'Kick Sauber': 6.5,
            'Haas': 4.5, 'Cadillac': 5.0
        }
    
    def _init_power_units(self):
        return {
            'Ferrari': {
                'teams': ['Ferrari', 'Haas', 'Cadillac'],
                'electric_rating': 8.5, 'reliability': 7.5
            },
            'Mercedes': {
                'teams': ['Mercedes', 'McLaren', 'Williams', 'Alpine'],
                'electric_rating': 9.0, 'reliability': 9.0
            },
            'Honda': {
                'teams': ['Aston Martin', 'RB'],
                'electric_rating': 8.0, 'reliability': 7.0
            },
            'Red Bull Ford': {
                'teams': ['Red Bull Racing'],
                'electric_rating': 7.5, 'reliability': 6.5
            },
            'Audi': {
                'teams': ['Kick Sauber'],
                'electric_rating': 7.0, 'reliability': 6.0
            }
        }
    
    def _init_track_data(self):
        return {
            'Bahrain': {'power': 0.7, 'downforce': 0.5},
            'Saudi Arabia': {'power': 0.8, 'downforce': 0.6},
            'Australia': {'power': 0.6, 'downforce': 0.6},
            'Japan': {'power': 0.5, 'downforce': 0.8},
            'China': {'power': 0.7, 'downforce': 0.5},
            'Miami': {'power': 0.6, 'downforce': 0.5},
            'Imola': {'power': 0.5, 'downforce': 0.7},
            'Monaco': {'power': 0.2, 'downforce': 0.9},
            'Canada': {'power': 0.7, 'downforce': 0.5},
            'Spain': {'power': 0.6, 'downforce': 0.7},
            'Austria': {'power': 0.8, 'downforce': 0.4},
            'Great Britain': {'power': 0.7, 'downforce': 0.7},
            'Hungary': {'power': 0.3, 'downforce': 0.9},
            'Belgium': {'power': 0.9, 'downforce': 0.5},
            'Netherlands': {'power': 0.6, 'downforce': 0.7},
            'Italy': {'power': 0.9, 'downforce': 0.4},
            'Azerbaijan': {'power': 0.8, 'downforce': 0.5},
            'Singapore': {'power': 0.3, 'downforce': 0.9},
            'United States': {'power': 0.6, 'downforce': 0.7},
            'Mexico': {'power': 0.7, 'downforce': 0.6},
            'Brazil': {'power': 0.7, 'downforce': 0.6},
            'Las Vegas': {'power': 0.8, 'downforce': 0.4},
            'Qatar': {'power': 0.7, 'downforce': 0.6},
            'Abu Dhabi': {'power': 0.7, 'downforce': 0.5}
        }
    
    def analyze_2024_2025_performance(self):
        """
        CRITICAL: Analyze 2024 and 2025 seasons for team/driver form
        This is the MAIN predictor (70% weight)
        """
        print("\n🔍 ANALYZING 2024-2025 PERFORMANCE (PRIMARY DATA)")
        print("=" * 70)
        
        all_results_2024 = []
        all_results_2025 = []
        
        # Load 2024 season
        print("\n📊 Loading 2024 season...")
        results_2024 = self._load_season_performance(2024)
        if not results_2024.empty:
            all_results_2024 = results_2024
            print(f"  ✓ Loaded {len(results_2024)} results from 2024")
        
        # Load 2025 season (if available)
        print("\n📊 Loading 2025 season...")
        results_2025 = self._load_season_performance(2025)
        if not results_2025.empty:
            all_results_2025 = results_2025
            print(f"  ✓ Loaded {len(results_2025)} results from 2025")
        else:
            print("  ⚠️  2025 data not yet available (pre-season)")
        
        # Calculate team performance scores
        print("\n📈 Calculating team performance scores...")
        self._calculate_team_scores(all_results_2024, all_results_2025)
        
        # Calculate driver performance scores
        print("👤 Calculating driver performance scores...")
        self._calculate_driver_scores(all_results_2024, all_results_2025)
        
        # Calculate team trajectory (improvement/decline)
        print("📉 Calculating team development trajectory...")
        self._calculate_team_trajectory(all_results_2024, all_results_2025)
        
        return all_results_2024, all_results_2025
    
    def _load_season_performance(self, year):
        """Load all race results from a season"""
        all_results = []
        
        try:
            schedule = fastf1.get_event_schedule(year)
            
            for idx, event in schedule.iterrows():
                if event['EventFormat'] != 'conventional':
                    continue
                
                try:
                    race = fastf1.get_session(year, event['EventName'], 'R')
                    race.load()
                    
                    for idx, row in race.results.iterrows():
                        all_results.append({
                            'year': year,
                            'race': event['EventName'],
                            'driver': row['Abbreviation'],
                            'team': row['TeamName'],
                            'position': row['Position'] if pd.notna(row['Position']) else 20,
                            'points': row['Points'] if pd.notna(row['Points']) else 0,
                            'grid': row['GridPosition'] if pd.notna(row['GridPosition']) else 20
                        })
                    
                except Exception as e:
                    pass  # Skip races that fail to load
            
        except Exception as e:
            print(f"  ⚠️  Could not load {year} season: {e}")
        
        return pd.DataFrame(all_results)
    
    def _calculate_team_scores(self, df_2024, df_2025):
        """
        Calculate team performance scores - HIGHER WEIGHT on recent performance
        """
        # Combine both seasons with 2025 weighted heavier
        if not df_2025.empty:
            # 2025 exists - weight it 70%, 2024 30%
            combined = pd.concat([df_2024, df_2025])
            combined['weight'] = combined['year'].map({2024: 0.3, 2025: 0.7})
        else:
            # No 2025 data - use 2024 100%
            combined = df_2024.copy()
            combined['weight'] = 1.0
        
        # Calculate weighted average position per team
        combined['weighted_pos'] = combined['position'] * combined['weight']
        team_avg_pos = combined.groupby('team')['weighted_pos'].sum() / combined.groupby('team')['weight'].sum()
        
        # Convert to score (lower position = higher score)
        # Position 1 = 10.0, Position 10 = 5.0, Position 20 = 0.0
        self.team_2025_performance = {}
        for team, avg_pos in team_avg_pos.items():
            score = max(0, 10 - (avg_pos - 1) * 0.5)
            self.team_2025_performance[team] = score
        
        print("\n  Team Performance Scores (2024-2025):")
        sorted_teams = sorted(self.team_2025_performance.items(), key=lambda x: x[1], reverse=True)
        for team, score in sorted_teams[:5]:
            print(f"    {team:<25} → {score:.2f}/10")
    
    def _calculate_driver_scores(self, df_2024, df_2025):
        """Calculate individual driver form"""
        if not df_2025.empty:
            combined = pd.concat([df_2024, df_2025])
            combined['weight'] = combined['year'].map({2024: 0.3, 2025: 0.7})
        else:
            combined = df_2024.copy()
            combined['weight'] = 1.0
        
        combined['weighted_pos'] = combined['position'] * combined['weight']
        driver_avg_pos = combined.groupby('driver')['weighted_pos'].sum() / combined.groupby('driver')['weight'].sum()
        
        self.driver_2025_performance = {}
        for driver, avg_pos in driver_avg_pos.items():
            score = max(0, 10 - (avg_pos - 1) * 0.5)
            self.driver_2025_performance[driver] = score
        
        print("\n  Top Driver Form:")
        sorted_drivers = sorted(self.driver_2025_performance.items(), key=lambda x: x[1], reverse=True)
        for driver, score in sorted_drivers[:5]:
            print(f"    {driver:<8} → {score:.2f}/10")
    
    def _calculate_team_trajectory(self, df_2024, df_2025):
        """
        Calculate if teams are improving or declining (2024 → 2025)
        CRITICAL for 2026 predictions!
        """
        if df_2025.empty:
            print("  ⚠️  No 2025 data - cannot calculate trajectory")
            self.team_trajectory = {team: 0.0 for team in self.team_2025_performance.keys()}
            return
        
        # Average position by team per year
        avg_2024 = df_2024.groupby('team')['position'].mean()
        avg_2025 = df_2025.groupby('team')['position'].mean()
        
        self.team_trajectory = {}
        for team in set(avg_2024.index) & set(avg_2025.index):
            # Negative = improving (lower positions), Positive = declining
            change = avg_2025[team] - avg_2024[team]
            # Convert to trajectory score
            trajectory = -change  # Reverse so positive = improving
            self.team_trajectory[team] = trajectory
        
        print("\n  Team Development Trajectory (2024→2025):")
        sorted_traj = sorted(self.team_trajectory.items(), key=lambda x: x[1], reverse=True)
        for team, traj in sorted_traj[:5]:
            arrow = "📈" if traj > 0 else "📉"
            print(f"    {arrow} {team:<25} → {traj:+.2f} positions")
    
    def calculate_2026_team_score(self, team, track):
        """
        ENHANCED scoring: 70% recent form, 30% regulation adaptation
        """
        # 1. Recent performance (70% weight) - PRIMARY FACTOR
        recent_score = self.team_2025_performance.get(team, 5.0) * 0.7
        
        # 2. Team trajectory bonus/penalty
        trajectory_bonus = self.team_trajectory.get(team, 0.0) * 0.1
        
        # 3. Regulation adaptation (30% weight) - SECONDARY FACTOR
        regulation_score = self.regulation_adaptation_scores.get(team, 5.0) * 0.3
        
        # 4. Power unit rating for track
        pu_rating = 5.0
        for pu_name, pu_data in self.power_unit_config.items():
            if team in pu_data['teams']:
                pu_rating = pu_data['electric_rating']
                break
        
        track_data = self.track_characteristics.get(track, {'power': 0.5, 'downforce': 0.5})
        pu_contribution = pu_rating * track_data['power'] * 0.2
        
        # Combine all factors
        total_score = recent_score + trajectory_bonus + regulation_score + pu_contribution
        
        return total_score
    
    def train_model(self, training_data, track_name):
        """
        Train ML model on combined 2024-2025 data
        """
        print("\n🎓 Training prediction model...")
        
        df = training_data.copy()
        
        # Add enhanced features
        df['team_recent_score'] = df['team'].map(self.team_2025_performance).fillna(5.0)
        df['team_trajectory'] = df['team'].map(self.team_trajectory).fillna(0.0)
        df['regulation_score'] = df['team'].map(self.regulation_adaptation_scores).fillna(5.0)
        
        # Add PU rating
        df['pu_rating'] = df['team'].apply(self._get_pu_rating)
        
        # Track characteristics
        track_data = self.track_characteristics.get(track_name, {'power': 0.5, 'downforce': 0.5})
        df['track_power'] = track_data['power']
        df['track_downforce'] = track_data['downforce']
        
        # Clean data
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['grid'] = pd.to_numeric(df['grid'], errors='coerce').fillna(20)
        df = df[df['position'].notna()]
        
        feature_cols = [
            'grid', 'team_recent_score', 'team_trajectory',
            'regulation_score', 'pu_rating', 'track_power', 'track_downforce'
        ]
        
        X = df[feature_cols]
        y = df['position']
        
        # Drop any NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print("  ⚠️  Insufficient training data!")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"  Training R² Score: {train_score:.3f}")
        print(f"  Testing R² Score: {test_score:.3f}")
        
        from sklearn.metrics import mean_absolute_error
        predictions = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        print(f"  Mean Absolute Error: {mae:.2f} positions")
        
        # Show feature importance
        print("\n  Feature Importance:")
        importances = self.model.feature_importances_
        for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {feat:<20} → {imp*100:.1f}%")
        
        return mae
    
    def _get_pu_rating(self, team):
        for pu_name, pu_info in self.power_unit_config.items():
            if team in pu_info['teams']:
                return pu_info['electric_rating']
        return 5.0
    
    def predict_2026_race(self, quali_data, track_name, uncertainty=0.30):
        """
        Predict race with enhanced 2024-2025 weighted scoring
        """
        print(f"\n🔮 Predicting 2026 {track_name} GP...")
        
        df = quali_data.copy()
        
        # Add all features
        df['team_recent_score'] = df['team'].map(self.team_2025_performance).fillna(5.0)
        df['team_trajectory'] = df['team'].map(self.team_trajectory).fillna(0.0)
        df['regulation_score'] = df['team'].map(self.regulation_adaptation_scores).fillna(5.0)
        df['pu_rating'] = df['team'].apply(self._get_pu_rating)
        
        track_data = self.track_characteristics.get(track_name, {'power': 0.5, 'downforce': 0.5})
        df['track_power'] = track_data['power']
        df['track_downforce'] = track_data['downforce']
        
        df['grid'] = pd.to_numeric(df.get('grid_position', df.get('quali_position', 10)), errors='coerce').fillna(10)
        
        feature_cols = [
            'grid', 'team_recent_score', 'team_trajectory',
            'regulation_score', 'pu_rating', 'track_power', 'track_downforce'
        ]
        
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        # Add uncertainty
        noise = np.random.normal(0, uncertainty * 3, len(predictions))
        predictions_adj = np.clip(predictions + noise, 1, 22)
        
        df['predicted_position'] = predictions_adj
        df['confidence'] = 1 - uncertainty
        df['2026_team_score'] = df['team'].apply(lambda t: self.calculate_2026_team_score(t, track_name))
        
        df = df.sort_values('predicted_position')
        
        return df[['driver', 'team', 'predicted_position', '2026_team_score', 'team_recent_score', 'confidence']]


# MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 70)
    print("🏎️  F1 2026 PREDICTION MODEL - ENHANCED")
    print("   70% Recent Form (2024-2025) + 30% Regulation Adaptation")
    print("=" * 70)
    
    predictor = F1_2026_Predictor_Enhanced()
    
    # PHASE 1: Analyze 2024-2025 performance (CRITICAL!)
    results_2024, results_2025 = predictor.analyze_2024_2025_performance()
    
    # Combine for training
    if not results_2025.empty:
        training_data = pd.concat([results_2024, results_2025])
        print(f"\n✓ Using {len(results_2024)} results from 2024 + {len(results_2025)} from 2025")
    else:
        training_data = results_2024
        print(f"\n✓ Using {len(results_2024)} results from 2024 only")
    
    # PHASE 2: Train model
    if not training_data.empty:
        mae = predictor.train_model(training_data, 'Australia')
    
    # PHASE 3: Make 2026 prediction
    print("\n🏁 Generating 2026 Australian GP Prediction...")
    
    quali_2026 = pd.DataFrame({
        'driver': ['VER', 'LEC', 'NOR', 'PIA', 'HAM', 'RUS', 'ALO', 'STR', 'SAI', 'HUL', 'BOT'],
        'team': ['Red Bull Racing', 'Ferrari', 'McLaren', 'McLaren', 'Ferrari', 'Mercedes', 
                 'Aston Martin', 'Aston Martin', 'Williams', 'Haas', 'Kick Sauber'],
        'quali_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'grid_position': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    })
    
    predictions = predictor.predict_2026_race(quali_2026, 'Australia', uncertainty=0.35)
    
    print("\n" + "=" * 80)
    print("🏆 2026 AUSTRALIAN GP PREDICTION (2024-2025 WEIGHTED MODEL)")
    print("=" * 80)
    print(f"{'Pos':<5} {'Driver':<8} {'Team':<25} {'2026 Score':<12} {'Recent Form':<12} {'Conf':<8}")
    print("-" * 80)
    
    for idx, row in predictions.head(11).iterrows():
        print(f"{int(row['predicted_position']):<5} {row['driver']:<8} {row['team']:<25} "
              f"{row['2026_team_score']:.2f}/10{'':<6} {row['team_recent_score']:.2f}/10{'':<6} {row['confidence']*100:.0f}%")
    
    print("\n" + "=" * 80)
    print("📊 MODEL WEIGHTS")
    print("=" * 80)
    print("• 2024-2025 Recent Performance: 70% (PRIMARY FACTOR)")
    print("• Team Development Trajectory:  10% (Improving vs Declining)")
    print("• Regulation Adaptation Score:  30% (Historical pattern)")
    print("• Power Unit & Track Fit:       20% (2026 specific)")
    print("\n💡 This model prioritizes RECENT FORM over historical regulation changes!")
    print("=" * 80)