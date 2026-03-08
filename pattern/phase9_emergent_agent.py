"""
Phase 9: The Emergent Agent
Give the system random structure + survival pressure
Watch it grow its own intelligence
"""

import numpy as np
import random


class EmergentAgent:
    """Emergent Agent - no structure, only survival pressure"""
    
    def __init__(self, n_units=20):
        # Random "neural soup"
        self.units = []
        for _ in range(n_units):
            self.units.append({
                'weights': np.random.randn(10) * 0.1,
                'energy': 1.0,
                'age': 0,
            })
        
        self.energy_pool = 100
        self.history = []
    
    def step(self, x, env):
        # Each unit tries to predict
        predictions = []
        
        for i, unit in enumerate(self.units):
            pred = np.dot(unit['weights'], x)
            predictions.append(pred)
            
            # Cost energy to compute
            unit['energy'] -= 0.01
            unit['age'] += 1
        
        # Actual value
        actual = env.generate()
        
        # Evaluate: prediction vs actual
        errors = [abs(p - actual) for p in predictions]
        
        min_error = min(errors)
        
        # Reward: good prediction = energy
        if min_error < 0.5:
            self.energy_pool += 1
        
        # Reallocate energy
        for i, unit in enumerate(self.units):
            if errors[i] == min_error and min_error < 0.5:
                unit['energy'] += 0.5
        
        # Death: zero energy = die
        dead = [i for i, u in enumerate(self.units) if u['energy'] < 0]
        
        # Birth: high energy = replicate
        born = []
        for i, u in enumerate(self.units):
            if u['energy'] > 2.0:
                new_unit = {
                    'weights': u['weights'] + np.random.randn(10) * 0.05,
                    'energy': u['energy'] * 0.5,
                    'age': 0,
                }
                born.append(new_unit)
                u['energy'] *= 0.5
        
        # Update
        for d in sorted(dead, reverse=True):
            del self.units[d]
        
        self.units.extend(born)
        
        # Record
        self.history.append({
            'n_units': len(self.units),
            'avg_energy': np.mean([u['energy'] for u in self.units]),
            'min_error': min_error
        })
        
        return min_error


def main():
    print('='*60)
    print('Phase 9: The Emergent Agent')
    print('='*60)
    print('\nStarting with random neural soup...')
    print('Survival pressure:')
    print('  - Computing costs energy')
    print('  - Good predictions gain energy')
    print('  - Low energy = death')
    print('  - High energy = reproduction\n')
    
    # Environment: periodic signal
    class PeriodicEnv:
        def __init__(self):
            self.phase = 0
        def generate(self):
            self.phase += 1
            return np.sin(2 * np.pi * self.phase / 10)
    
    agent = EmergentAgent(n_units=20)
    env = PeriodicEnv()
    
    # Run
    for step in range(500):
        x = np.random.randn(10)
        error = agent.step(x, env)
        
        if step % 100 == 0:
            print(f'Step {step}: units={len(agent.units)}, error={error:.3f}')
    
    # Result
    print('\n=== Final State ===')
    print(f'Units: {len(agent.units)}')
    
    if len(agent.units) > 0:
        ages = [u['age'] for u in agent.units]
        print(f'Max age: {max(ages)}')
        
        weights = [u['weights'] for u in agent.units]
        weight_variance = np.var([np.mean(w) for w in weights])
        print(f'Weight variance: {weight_variance:.4f}')
        
        if weight_variance > 0.01:
            print('[OK] Structure differentiation detected!')
        else:
            print('[WARN] No differentiation')


if __name__ == "__main__":
    main()
