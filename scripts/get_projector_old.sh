#!/usr/bin/sh

if [ -f $1 ]; then
    YAML=$1
else
    cat << EOF
ERROR: Need a \`list.yaml\` file
 -> Usage: get_projector.sh [<YAML file>]
EOF
    exit 1
fi

MODEL="ViT-L/14"        # Model to extract the text embeddings (for ViT models it uses the text encoder of CLIP)
REDUCTION="pca"         # Reduction method
DIMENSION=384           # The dimension to which reduce the embedding with the `REDUCTION`
CENTROID="mean"         # The method to compute the centroids
TRUNCATE=1              # If the input to the model should be truncated 

cat << EOF
Generating projector with:
 -> Model: $MODEL
 -> Reduction method: $REDUCTION
 -> Dimension: $DIMENSION
 -> Centroid: $CENTROID
 -> Truncate: $TRUNCATE
EOF

python projector.py --yaml $YAML --model $MODEL --reduction-method $REDUCTION \
    --dimension $DIMENSION --centroid-method $CENTROID --truncate-input $TRUNCATE