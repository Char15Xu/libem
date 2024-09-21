from libem.optimize.interface import get_openai_cost

def low_confidence_filter(result, threshold=0.5):
    low_confidence_pairs = []
    high_confidence_pairs = []

    for match in result.get('match', []):
        confidence = match.get('confidence')
        if confidence is not None and confidence < threshold:
            low_confidence_pairs.append({
                'left': match['left'],
                'right': match['right'],
                'label': match['label']
            })
        else:
            high_confidence_pairs.append(match)

    updated_results = result.copy()
    updated_results['match'] = high_confidence_pairs

    return low_confidence_pairs, updated_results
