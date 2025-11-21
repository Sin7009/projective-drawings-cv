# Technical Specification: Wartegg Drawing Test (WZT) Logic

## 1. Core Philosophy
The project automates the "Modified Wartegg Drawing Test" (Valyavko & Knyazev, 2014). Unlike standard approaches, we treat drawings as data points to measure latent psychological variables (Anxiety, Aggression, Social Contact).

## 2. The Wartegg (WZT) Logic Mapping
Based on the papers, here is the logic for the 8 squares that needs to be implemented in code:

| ROI (Region) | Stimulus (Archetype) | Psychological Meaning [Source: Valyavko & Knyazev] | CV Feature to Extract |
| :--- | :--- | :--- | :--- |
| **Square 1** | Central Point | **Self-Confidence / Ego** | **Centroid Analysis:** Did the user draw *around* the point (centering = confidence) or ignore/overwhelm it? |
| **Square 2** | Wavy Line | **Empathy / Social Contact** | **Curve Analysis:** Measure "smoothness" vs. "sharpness". Sharp angles = emotional coldness. Flowing lines = contactness. |
| **Square 3** | 3 Rising Lines | **Ambition / Achievement** | **Trend Regression:** Calculate the slope of drawn lines. Upward trend = ambition. Downward/Flat = lack of motivation. |
| **Square 4** | Black Square | **Anxiety / Fears** | **Pixel Density:** Analyze shading intensity around the black square. Heavy darkening = high anxiety. |
| **Square 5** | Opposing Lines | **Aggression / Activity** | **Intersection Detection:** Do the lines connect (overcoming obstacles) or stay apart (passivity)? Detect sharp spikes (aggression). |
| **Square 6** | Horizontal/Vertical Lines | **Rationality / Integration** | **Gestalt Closure:** Do the lines form a closed geometric object (synthesis)? Or remain disjointed? |
| **Square 7** | Dotted Semicircle | **Sensitivity / Tact** | **Texture Continuity:** Did the user keep the dotted texture (sensitivity) or overwrite it with solid lines (bluntness)? |
| **Square 8** | Large Arc | **Protection / Social Norms** | **Enclosure:** Is the arc used as a roof/umbrella (protection) or something open? |

## 3. Validation Ground Truth
We will validate our CV models against the correlation data found in the papers:
- **Anxiety:** Must correlate ~0.7 with "Temml-Dorki-Amen Test".
- **Aggression:** Must correlate with observation data.
- **Social Contact:** Must correlate with "Family Drawing" metrics.
