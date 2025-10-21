#!/usr/bin/env python3
"""
Test script to verify that the gym_sa package can be imported correctly.
"""

def test_imports():
    """Test that all main components can be imported."""
    try:
        # Test main package import
        import gym_sa
        print("✓ Successfully imported gym_sa package")
        print(f"  Version: {gym_sa.__version__}")

        # Test main classes
        from gym_sa import Annealer, ParallelAnnealer, TSPEnv
        print("✓ Successfully imported main classes")

        # Test utility functions
        from gym_sa import softmax, set_seed
        print("✓ Successfully imported utility functions")

        # Test logger
        from gym_sa import CSVLogger
        print("✓ Successfully imported CSVLogger")

        # Test benchmark modules
        from gym_sa import tsp_benchmark, psa_benchmark
        print("✓ Successfully imported benchmark modules")

        print("\n🎉 All imports successful! The package is ready to use.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the package."""
    try:
        import numpy as np
        from gym_sa import TSPEnv, Annealer

        # Test TSP environment creation
        env = TSPEnv(n_cities=5, seed=42)
        print("✓ Successfully created TSP environment")

        # Test Annealer creation
        annealer = Annealer(
            env_constructor=TSPEnv,
            env_params={"n_cities": 5, "seed": 42},
            initial_temperature=1.0,
            cooling_rate=0.99
        )
        print("✓ Successfully created Annealer")

        print("🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing gym_sa package...\n")

    import_success = test_imports()

    if import_success:
        print("\nTesting basic functionality...")
        func_success = test_basic_functionality()

        if func_success:
            print("\n✅ Package is fully functional!")
        else:
            print("\n⚠️  Package imports work but functionality needs attention.")
    else:
        print("\n❌ Package import failed. Please check the installation.")
