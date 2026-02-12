try:
    import pygame
    import moderngl
    import numpy
    import gymnasium
    import glcontext
    print("All packages imported successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
