def get_stress_level(stress_index: float) -> str:
    """Takes a stress index from 0 to 1 and returns the categorical level."""
    if stress_index < 0.3:
        return "Low"
    elif stress_index < 0.6:
        return "Moderate"
    elif stress_index < 0.8:
        return "High"
    return "Severe"

def generate_stress_description(thermal: float, water: float) -> str:
    """Provides a human-readable explanation based on stress components."""
    if thermal < 0.3 and water < 0.3:
        return "Growing conditions were generally favorable with minimal stress detected."
        
    parts = []
    if thermal >= 0.6:
        parts.append("severe thermal stress")
    elif thermal >= 0.3:
        parts.append("moderate thermal stress")
        
    if water >= 0.6:
        parts.append("severe water deficit")
    elif water >= 0.3:
        parts.append("moderate water deficit")
        
    if len(parts) == 2:
        return f"Crop experienced both {parts[0]} and {parts[1]}."
    elif len(parts) == 1:
        return f"Crop experienced {parts[0]}, though other conditions were favorable."
        
    return "Mixed environmental stressors detected."
