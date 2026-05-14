"""Short, plain-language disease notes shown alongside predictions."""

ADVISORIES: dict[str, dict[str, str]] = {
    "Tomato___Bacterial_spot": {
        "cause": "Caused by Xanthomonas bacteria; spreads in warm, wet conditions.",
        "symptoms": "Small dark water-soaked spots on leaves, later turning brown with yellow halos.",
        "action": "Remove infected leaves, avoid overhead watering, apply copper-based bactericides.",
    },
    "Tomato___Early_blight": {
        "cause": "Fungus Alternaria solani; favored by warm, humid weather and crop stress.",
        "symptoms": "Brown concentric rings on older leaves with a yellow halo.",
        "action": "Prune affected leaves, mulch the soil, rotate crops, apply chlorothalonil or mancozeb.",
    },
    "Tomato___Late_blight": {
        "cause": "Oomycete Phytophthora infestans; spreads rapidly in cool, wet conditions.",
        "symptoms": "Greasy, water-soaked patches on leaves with white mold underneath.",
        "action": "Destroy infected plants immediately, improve airflow, apply copper or mancozeb fungicides.",
    },
    "Tomato___Leaf_Mold": {
        "cause": "Fungus Passalora fulva; common in greenhouses with poor ventilation.",
        "symptoms": "Yellow patches on top of leaves, olive-green mold underneath.",
        "action": "Improve ventilation, reduce humidity, apply approved fungicides if severe.",
    },
    "Tomato___Septoria_leaf_spot": {
        "cause": "Fungus Septoria lycopersici; spreads in wet weather and dense foliage.",
        "symptoms": "Many small circular spots with dark borders and gray centers on lower leaves.",
        "action": "Remove infected lower leaves, stake plants for airflow, apply copper-based fungicides.",
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "cause": "Two-spotted spider mite infestation; common in hot, dry weather.",
        "symptoms": "Tiny yellow speckles on leaves, fine webbing, and bronzing as damage worsens.",
        "action": "Spray water on undersides of leaves, introduce predatory mites, use insecticidal soap.",
    },
    "Tomato___Target_Spot": {
        "cause": "Fungus Corynespora cassiicola; spreads in warm, humid weather.",
        "symptoms": "Brown leaf spots with concentric rings, similar to a target.",
        "action": "Prune to improve airflow, rotate crops, apply protective fungicides.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Virus spread by whiteflies; no chemical cure once infected.",
        "symptoms": "Upward curling leaves, yellowing, stunted growth, reduced fruit set.",
        "action": "Remove infected plants, control whiteflies with yellow sticky traps or insecticides.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "cause": "Tobamovirus; spreads via contact, tools, and seeds.",
        "symptoms": "Mottled light and dark green pattern on leaves; distorted growth.",
        "action": "Destroy infected plants, sterilize tools, plant resistant varieties.",
    },
    "Tomato___healthy": {
        "cause": "No disease detected.",
        "symptoms": "Leaves appear uniformly green and undamaged.",
        "action": "Maintain good practices: balanced watering, crop rotation, and routine inspection.",
    },
}


def advisory_for(class_id: str) -> dict[str, str]:
    return ADVISORIES.get(
        class_id,
        {
            "cause": "Unknown.",
            "symptoms": "No advisory information available for this class.",
            "action": "Consult an agricultural extension officer for confirmation.",
        },
    )
