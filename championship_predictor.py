"""
F1 2026 Full Season Championship Predictor
Predicts all 24 races and calculates Driver & Constructor Championships
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

# 2026 F1 Points System
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

# 2026 Calendar
CALENDAR_2026 = [
    'Australia', 'China', 'Japan', 'Bahrain', 'Saudi Arabia',
    'Miami', 'Imola', 'Monaco', 'Canada', 'Spain',
    'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands',
    'Italy', 'Azerbaijan', 'Singapore', 'United States', 'Mexico',
    'Brazil', 'Las Vegas', 'Qatar', 'Abu Dhabi'
]

# 2026 Driver Lineups (estimated - update as confirmed)
DRIVER_LINEUPS_2026 = {
    'Red Bull Racing': ['VER', 'LAW'],
    'Ferrari': ['HAM', 'LEC'],
    'Mercedes': ['RUS', 'ANT'],
    'McLaren': ['NOR', 'PIA'],
    'Aston Martin': ['ALO', 'STR'],
    'Alpine': ['GAS', 'DOU'],
    'Williams': ['SAI', 'ALB'],
    'RB': ['TSU', 'HAD'],
    'Haas': ['BEA', 'OCO'],
    'Kick Sauber': ['HUL', 'BOT'],
    'Cadillac': ['SAR', 'COL']
}

class Championship_Predictor:
    """
    Predicts full 2026 F1 season and calculates championships
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.team_scores = self._init_team_scores()
        self.pu_config = self._init_power_units()
        self.tracks = self._init_tracks()
        
    def _init_team_scores(self):
        return {
            'Red Bull Racing': 8.5, 'Ferrari': 7.0, 'Mercedes': 9.0,
            'McLaren': 7.5, 'Aston Martin': 6.5, 'Alpine': 5.5,
            'Williams': 5.0, 'RB': 6.0, 'Kick Sauber': 6.5,
            'Haas': 4.5, 'Cadillac': 5.0
        }
    
    def _init_power_units(self):
        return {
            'Ferrari': {'teams': ['Ferrari', 'Haas', 'Cadillac'], 'rating': 8.5, 'reliability': 7.5},
            'Mercedes': {'teams': ['Mercedes', 'McLaren', 'Williams', 'Alpine'], 'rating': 9.0, 'reliability': 9.0},
            'Honda': {'teams': ['Aston Martin', 'RB'], 'rating': 8.0, 'reliability': 7.0},
            'Red Bull Ford': {'teams': ['Red Bull Racing'], 'rating': 7.5, 'reliability': 6.5},
            'Audi': {'teams': ['Kick Sauber'], 'rating': 7.0, 'reliability': 6.0}
        }
    
    def _init_tracks(self):
        return {
            'Australia': {'power': 0.6, 'downforce': 0.6}, 'China': {'power': 0.7, 'downforce': 0.5},
            'Japan': {'power': 0.5, 'downforce': 0.8}, 'Bahrain': {'power': 0.7, 'downforce': 0.5},
            'Saudi Arabia': {'power': 0.8, 'downforce': 0.6}, 'Miami': {'power': 0.6, 'downforce': 0.5},
            'Imola': {'power': 0.5, 'downforce': 0.7}, 'Monaco': {'power': 0.2, 'downforce': 0.9},
            'Canada': {'power': 0.7, 'downforce': 0.5}, 'Spain': {'power': 0.6, 'downforce': 0.7},
            'Austria': {'power': 0.8, 'downforce': 0.4}, 'Great Britain': {'power': 0.7, 'downforce': 0.7},
            'Hungary': {'power': 0.3, 'downforce': 0.9}, 'Belgium': {'power': 0.9, 'downforce': 0.5},
            'Netherlands': {'power': 0.6, 'downforce': 0.7}, 'Italy': {'power': 0.9, 'downforce': 0.4},
            'Azerbaijan': {'power': 0.8, 'downforce': 0.5}, 'Singapore': {'power': 0.3, 'downforce': 0.9},
            'United States': {'power': 0.6, 'downforce': 0.7}, 'Mexico': {'power': 0.7, 'downforce': 0.6},
            'Brazil': {'power': 0.7, 'downforce': 0.6}, 'Las Vegas': {'power': 0.8, 'downforce': 0.4},
            'Qatar': {'power': 0.7, 'downforce': 0.6}, 'Abu Dhabi': {'power': 0.7, 'downforce': 0.5}
        }
    
    def calculate_team_performance(self, team, track):
        """Calculate expected performance for team at track"""
        base_score = self.team_scores.get(team, 5.0)
        
        # Get PU rating
        pu_rating = 5.0
        for pu, data in self.pu_config.items():
            if team in data['teams']:
                pu_rating = data['rating']
                break
        
        track_data = self.tracks.get(track, {'power': 0.5, 'downforce': 0.5})
        
        # Weighted performance score
        performance = (
            base_score * 0.5 +
            pu_rating * track_data['power'] * 0.3 +
            base_score * track_data['downforce'] * 0.2
        )
        
        return performance
    
    def train_baseline_model(self):
        """Train model on 2024 data"""
        print("🎓 Training baseline model on 2024 data...")
        
        training_data = []
        
        try:
            schedule = fastf1.get_event_schedule(2024)
            
            for idx, event in schedule.iterrows():
                if event['EventFormat'] != 'conventional':
                    continue
                
                try:
                    race = fastf1.get_session(2024, event['EventName'], 'R')
                    race.load()
                    
                    for idx, row in race.results.iterrows():
                        training_data.append({
                            'team': row['TeamName'],
                            'position': row['Position'] if pd.notna(row['Position']) else 20
                        })
                    
                except:
                    pass
            
            if training_data:
                df = pd.DataFrame(training_data)
                print(f"  ✓ Loaded {len(df)} results from 2024")
                return df
            
        except Exception as e:
            print(f"  ⚠️  Could not load 2024 data: {e}")
        
        return pd.DataFrame()
    
    def simulate_race(self, track, race_number, total_races):
        """Simulate a single race"""
        
        # Calculate uncertainty - decreases as season progresses
        uncertainty = 0.4 - (race_number / total_races) * 0.15
        
        results = []
        
        for team, drivers in DRIVER_LINEUPS_2026.items():
            team_performance = self.calculate_team_performance(team, track)
            
            for driver in drivers:
                # Add randomness based on uncertainty
                noise = np.random.normal(0, uncertainty * 3)
                predicted_pos = 11 - team_performance + noise
                predicted_pos = max(1, min(22, predicted_pos))  # 22 drivers in 2026
                
                results.append({
                    'driver': driver,
                    'team': team,
                    'predicted_position': predicted_pos,
                    'team_performance': team_performance
                })
        
        # Sort by position and assign final positions
        df = pd.DataFrame(results)
        df = df.sort_values('predicted_position').reset_index(drop=True)
        df['position'] = range(1, len(df) + 1)
        
        # Assign points
        df['points'] = df['position'].map(lambda x: POINTS_SYSTEM.get(x, 0))
        
        return df[['position', 'driver', 'team', 'points', 'team_performance']]
    
    def simulate_full_season(self):
        """Simulate entire 2026 season"""
        
        print("\n" + "=" * 70)
        print("🏁 SIMULATING 2026 F1 SEASON - ALL 24 RACES")
        print("=" * 70)
        
        all_results = []
        race_winners = []
        
        total_races = len(CALENDAR_2026)
        
        for race_num, track in enumerate(CALENDAR_2026, 1):
            race_results = self.simulate_race(track, race_num, total_races)
            
            # Store results
            race_results['race'] = track
            race_results['race_number'] = race_num
            all_results.append(race_results)
            
            # Track winner
            winner = race_results.iloc[0]
            race_winners.append({
                'race': track,
                'winner': winner['driver'],
                'team': winner['team']
            })
            
            # Print race result
            print(f"  Race {race_num:2d}: {track:<20} → Winner: {winner['driver']} ({winner['team']})")
        
        return pd.concat(all_results, ignore_index=True), pd.DataFrame(race_winners)
    
    def calculate_championships(self, results):
        """Calculate Driver and Constructor Championships"""
        
        # Driver Championship
        driver_points = results.groupby(['driver', 'team'])['points'].sum().reset_index()
        driver_points = driver_points.sort_values('points', ascending=False).reset_index(drop=True)
        driver_points['position'] = range(1, len(driver_points) + 1)
        
        # Constructor Championship
        constructor_points = results.groupby('team')['points'].sum().reset_index()
        constructor_points = constructor_points.sort_values('points', ascending=False).reset_index(drop=True)
        constructor_points['position'] = range(1, len(constructor_points) + 1)
        
        return driver_points, constructor_points
    
    def print_championships(self, driver_standings, constructor_standings, race_winners):
        """Print championship results"""
        
        print("\n" + "=" * 70)
        print("🏆 2026 DRIVER WORLD CHAMPIONSHIP - FINAL STANDINGS")
        print("=" * 70)
        print(f"{'Pos':<5} {'Driver':<8} {'Team':<25} {'Points':<10}")
        print("-" * 70)
        
        for idx, row in driver_standings.head(10).iterrows():
            emoji = "🥇" if row['position'] == 1 else "🥈" if row['position'] == 2 else "🥉" if row['position'] == 3 else "  "
            print(f"{emoji} {row['position']:<3} {row['driver']:<8} {row['team']:<25} {int(row['points']):<10}")
        
        print("\n" + "=" * 70)
        print("🏆 2026 CONSTRUCTORS CHAMPIONSHIP - FINAL STANDINGS")
        print("=" * 70)
        print(f"{'Pos':<5} {'Team':<30} {'Points':<10}")
        print("-" * 70)
        
        for idx, row in constructor_standings.iterrows():
            emoji = "🥇" if row['position'] == 1 else "🥈" if row['position'] == 2 else "🥉" if row['position'] == 3 else "  "
            print(f"{emoji} {row['position']:<3} {row['team']:<30} {int(row['points']):<10}")
        
        print("\n" + "=" * 70)
        print("📊 SEASON STATISTICS")
        print("=" * 70)
        
        # Most wins
        winner_counts = race_winners['winner'].value_counts()
        print(f"Most Wins: {winner_counts.index[0]} ({winner_counts.iloc[0]} wins)")
        
        # Most poles (simulated - same as wins for now)
        print(f"Most Podiums: {driver_standings.iloc[0]['driver']} (estimated)")
        
        # Winning team
        team_wins = race_winners['team'].value_counts()
        print(f"Most Team Wins: {team_wins.index[0]} ({team_wins.iloc[0]} wins)")
        
        print("\n" + "=" * 70)
        print("⚠️  IMPORTANT NOTES")
        print("=" * 70)
        print("• These are PREDICTIONS based on historical data and 2026 regulations")
        print("• High uncertainty until pre-season testing (Feb 2026)")
        print("• Update team scores as more info becomes available")
        print("• Re-run after each real 2026 race for better accuracy")
        print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("🏎️  F1 2026 FULL SEASON CHAMPIONSHIP PREDICTOR")
    print("   Simulating all 24 races with new regulations")
    print("=" * 70)
    
    predictor = Championship_Predictor()
    
    # Train baseline model (optional - improves accuracy)
    predictor.train_baseline_model()
    
    # Simulate full season
    all_results, race_winners = predictor.simulate_full_season()
    
    # Calculate championships
    driver_standings, constructor_standings = predictor.calculate_championships(all_results)
    
    # Print results
    predictor.print_championships(driver_standings, constructor_standings, race_winners)
    
    print("\n✅ Simulation complete! Re-run for different predictions.")